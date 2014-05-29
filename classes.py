import re
from lxml import etree
import os
import numpy
from collections import defaultdict, OrderedDict
from tags import *
import codecs
import copy


class Token(object):
    """ Class designed to encapsulate the idea of a token.  This includes
    the token itself,  plus pre and post whitespace,  as well as the start and
    end positions of the token with-in the document that the token was parsed
    out of.  It also includes an 'index' attribute that can be set by external
    functions and classes (see TokenSequence).
    """
    def __init__(self, token, pre_ws, post_ws, index, start, end):
        self.token = token
        self.start = int(start)
        self.end = int(end)
        self.index = int(index)

        # pre whitespace
        self.pre_ws = pre_ws
        # post whitespace
        self.post_ws = post_ws


    def __repr__(self):
        return "<{}: {}, {}, {}, i:{}, s:{}, e:{}>".format(self.__class__.__name__, 
                                                           self.pre_ws.__repr__(), 
                                                           self.token.__repr__(), 
                                                           self.post_ws.__repr__(), 
                                                           self.index,
                                                           self.start, self.end)

    def __str__(self):
        return self.token + self.post_ws
        
    def __len__(self):
        return len(str(self))

    def __hash__(self):
        return hash(self.start, self.end)

    def __eq__(self, other):
        """ Test the equality of two tokens. Based on start and end values. 
        """
        if other.start == self.start and other.end == self.end:
            return True

        return False



class TokenSequence(object):
    """ Encapsulates the functionality of a sequence of tokens.  it is designed
    to parse using the tokenizer() classmethod,  but can use any other subclassed
    method as long as it returns a list of Token() objects.
    """
    tokenizer_re = re.compile(r'(\w+)')
    
    @classmethod
    def tokenizer(cls, text, start=0):

        # This could be a one-liner,  but we'll split it up 
        # so its a litle clearer.

        # This generates a list of strings in the form 
        # [WHTEPSACE, TOKEN, WHITESPACE, TOKEN ...]
        split_tokens = re.split(cls.tokenizer_re, text)
        
        # This generates trigrams from the list in the form
        # [(WHITESPACE, TOKEN, WHITESPACE),
        #  (TOKEN, WHITESPACE, TOKEN),
        #  (WHITESPACE, TOKEN, WHITESPACE)
        #  .... ]
        token_trigrams =  zip(*[split_tokens[i:] for i in range(3)])

        # This keeps only odd tuples from token trigrams,  ie:
        # [(WHITESPACE, TOKEN, WHITESPACE),
        #  (WHITESPACE, TOKEN, WHITESPACE),
        #  (WHITESPACE, TOKEN, WHITESPACE)
        #  .... ]
        token_tuples = [t for i,t in enumerate(token_trigrams) if not bool(i & 1)]

        tokens = []
        index = 0

        # Add a dummy token that accounts for the leading whitespace
        # But only if we're starting at the begining of a document
        # If we're dealing with a tag or some other mid-document location
        # skip this part
        if start == 0:
            tokens.append(Token("", "", split_tokens[0], index, 0, 0))
            index += 1

        # Calculate start and end positions of the non-whitespace/punctuation
        # and append the token with its index into the list of tokens.
        for pre, token, post in token_tuples:
            token_start = start + len(pre)
            token_end = token_start + len(token)
            start = token_end
            tokens.append(Token(token, pre, post, index, token_start, token_end))
            index += 1
        


        return tokens


    def __init__(self, text, tokenizer=None, start=0):


        tokenizer = TokenSequence.tokenizer if tokenizer == None else tokenizer

        if hasattr(text, "__iter__"):            
            self.text = ''.join(str(t) for t in text)
            self.tokens = text
            
        else:
            self.text = text
            self.tokens = tokenizer(self.text, start=start)
    
        # If start is 0 we assume we're parsing a whole document 
        # and not a sub-string of tokens.
        if start == 0:
            assert len(self.text) == sum(len(t) for t in self.tokens), \
                "Tokenizer MUST return a list of strings with character " \
                "length equal to text length"



    def __str__(self):
        return ''.join([str(t).encode("string_escape") for t in self.tokens])


    def __repr__(self):
        return "<{} '{}', s:{}, e:{}>".format(self.__class__.__name__,
                                            str(self) if len(str(self)) < 40 else str(self)[:37] + "...",
                                            self[0].start,
                                            self[-1].end)


    def __len__(self):
        return len(self.tokens)


    def __getitem__(self, index):
        return self.tokens[index]


    def __iter__(self):
        return self.tokens.__iter__()
    
    def next(self):
        return self.tokens.next()


    def subseq(self, other):
        """Test if we are a subsequence of other"""
        return all([t in other.tokens for t in self.tokens])




class StandoffAnnotation(object):
    """ This class models a standoff annotation,  including parsing out file ID 
    information,  processing text and tags into objectsand coverting these
    objects back into XML elements,  dicts, files, token sequences etc.
    """
    id_parser = re.compile(r'^(\d+)-(\d+)(.*)\.xml')
    token_sequence_class = TokenSequence

    def __init__(self, file_name=None, root="root"):
        self.patient_id = ''
        self.record_id = ''
        self.sys_id = ''
        self.file_name = None
        self.raw = None
        self.text = None
        self.root = root
        self.doc_tags = []
        self.tags = []
        self.phi = []
        self._tokens = None

        if file_name:
            if self.id_parser.match(os.path.basename(file_name)):
                self.patient_id, \
                self.record_id, \
                self.sys_id = self.id_parser.match(os.path.basename(file_name))\
                                            .groups()
            else:
                self.patient_id = os.path.splitext(os.path \
                                                   .basename(file_name))[0]
        else:
            self.patient_id = None

            
        if file_name is not None:
            with open(file_name, 'r') as handle:
                self.parse_text_and_tags(handle.read().decode('utf8'))
                self.file_name = file_name
                
    @property
    def id(self):
        return self.patient_id + "-" + self.record_id

    @id.setter
    def id(self, value):
        self.patient_id, self.record_id = value.split("-")

    @property
    def token_sequence(self):
        if self._tokens == None:
            self._tokens = self.token_sequence_class(self.text, self.token_sequence_class.tokenizer)

        return self._tokens

    
    def tag_to_token_sequence(self, tag):
        try:
            seq = self.token_sequence_class(tag.text, start=int(tag.start))
            for token in seq:
                try:
                    token.index = self.token_sequence.tokens.index(token)
                except ValueError:
                    token.index = None
            return seq
        except:
            return []



    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id and other.id == self.id


    def get_annotation_tag_color(self, name):
        tag_colors = {"DIABETES" : ('\033[31m','\033[0m'),
                      "CAD" : ('\033[31m','\033[0m'),
                      "HYPERTENSION" : ('\033[31m','\033[0m'),
                      "HYPERLIPIDEMIA" : ('\033[31m','\033[0m'),
                      "SMOKER" : ('\033[31m','\033[0m'),
                      "OBESE" : ('\033[31m','\033[0m'),
                      "FAMILY_HIST" : ('\033[31m','\033[0m'),
                      "MEDICATION" : ('\033[31m','\033[0m')}

        try:
            return tag_colors[name]
        except KeyError:
            return ('\033[90m','\033[0m')


    def get_annotator_marked_text(self):
        """ Go through marking the entire text with color codes used to 
        identify different types of human annotations."""
        text = self.get_text()
        
        # This should ensure we have eliminated any overlapping positions 
        # information
        positions = []
        for tag in self.get_tags():
            if hasattr(tag, "start") and hasattr(tag, "end"):
                positions.append((tag.get_start(), tag.get_end(), tag))
        
        if len(positions):
            positions.sort(key=lambda x: x[0])

            last_start = positions[0][0]
            last_end = positions[0][1]
            concat = []
            for start,end,t in positions[1:]:
                if start <= last_end:
                    if end >= last_end:
                        last_end = end
                else:
                    concat.append((last_start,last_end, t))
                    last_start = start
                    last_end = end

            concat.append((last_start,last_end, t))

            # return the text
            for start,end,tag in sorted(concat, key=lambda x: x[0], reverse=True):
                open_str, close_str  = self.get_annotation_tag_color(tag.name)
                text = text[:start] + open_str + \
                       text[start:end] + close_str + text[end:]

        return text

    
    def toElement(self, with_phi_tags=True, 
                  with_annotator_tags=True, 
                  with_doc_level=True):

        root = etree.Element(self.root)
        text = etree.SubElement(root, "TEXT")
        tags = etree.SubElement(root, "TAGS")
        text.text = etree.CDATA(self.text)

        if with_doc_level:
            for t in self.doc_tags:                
                try:
                    tags.append(t.toElement(with_annotator_tags \
                                            = with_annotator_tags))
                # MAE convertion throws all tags into doc_tags, because regular
                # tags don't have the with_annotator_tags argument we need to 
                # catch and append the regular tag here.
                except TypeError:
                    tags.append(t.toElement())
        elif with_annotator_tags and not with_doc_level:
            for t in self.doc_tags:                
                for at in t.annotator_tags:
                    tags.append(at.toElement())
        
        if with_phi_tags == True:
            for t in self.get_phi():
                tags.append(t.toElement())
    
        return root

    def toListOfDicts(self, with_phi_tags=True, 
                      with_annotator_tags=True, 
                      with_doc_level=True, 
                      attrs=None):
        tag_list = []
        for t in self.get_doc_tags():
            if with_doc_level:
                tag_list.append(t.toDict(attributes=attrs))
            if with_annotator_tags:
                for at in t.annotator_tags:
                    tag_list.append(at.toDict(attributes=attrs))
        if with_phi_tags:
            for t in self.get_phi():
                tag_list.append(t.toDict(attributes=attrs))
        
        return tag_list
            
        

    def toXML(self, **kwargs):
        
        pretty_print = kwargs.pop("pretty_print", True)
        
        return etree.tostring(self.toElement(**kwargs), 
                              pretty_print=pretty_print, 
                              xml_declaration=True, encoding='UTF-8')

    def save(self, **kwargs):
        """ Save the standoff annotation to either self.file_name or to a 
        file defined by the key word argument "path." Accepts
          path
          pretty_print
          with_phi_tags
          with_annotator_tags
          with_doc_level
        keyword arguments. and passes those on to toXML before writing to
        file. 
        """

        path = kwargs.pop("path",self.file_name)

        if "pretty_print" not in kwargs:
            kwargs["pretty_print"] = True

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # MAE has some specific requirements for its XML parsing,  here
        # We remove all leading whitespace from tags, this could potentially
        # introduce bugs if information in the <TEXT></TEXT> element starts with
        # whitespace and the first character is a '<'
        if kwargs['pretty_print'] == "MAE":
            xml = self.toXML(**kwargs)
            
            with open(path,"w") as h:
                h.writelines([re.sub("^\s+<", "<", l + "\n") 
                              for l in xml.split("\n")])

        else:
            with open(path, "w") as h:
                h.write(self.toXML(**kwargs))
        
        return True
    
    def __repr__(self):
        return u"<StandoffAnnotation (%s) %s: tags:%s phi:%s>" \
        % (self.sys_id, self.id, len(self.get_tags()), len(self.get_phi()))

    def get_filename(self):
        return self.file_name

    def get_phi(self):
        return self.phi

    def get_text(self):
        return self.text

    def get_tag(self, ident):
        for t in self.get_tags():
            if t.id == ident:
                return t
        return None

    def get_doc_tags(self):
        if len(self.doc_tags) == 0:
            hash_dict = defaultdict(list)

            # hash our tags based on their document level annotation
            for a in self.get_tags():
                hash_dict[a.get_document_annotation()].append(a)

            # Give the document tags id's and make sure corrisponding annotator
            # tags are related to the correct document level tag through their
            # docid attribute.
            i = 0            
            for doc_tag, annotator_tags in hash_dict.items():

                doc_tag.id = "DOC%s" % i
                doc_tag.annotator_tags = annotator_tags

                self.doc_tags.append(doc_tag)
                i += 1

            self.doc_tags = self.doc_tags

        return self.doc_tags

            
    def get_tags(self):
        if len(self.tags) == 0:
            return [at for dt in self.doc_tags for at in dt.annotator_tags]
        else:
            return self.tags

    def get_sorted_tags(self,reverse=False):
        return sorted(self.get_tags(), key=\
                      lambda tag: tag.get_start(), reverse=reverse)

    def parse_text_and_tags(self, text=None):
        if text is not None:
            self.raw = text
        
        soup = etree.fromstring(self.raw.encode("utf8"))
        self.root = soup.tag    

        try:
            self.text = soup.find("TEXT").text
        except AttributeError:
            self.text = None

            
        # Handles files where PHI, and AnnotatorTags are all just 
        # stuffed into tag element.
        for t, cls in PHITag.tag_types.items():
            if len(soup.find("TAGS").findall(t)):
                for element in soup.find("TAGS").findall(t):
                    self.phi.append(cls(element))
   

        for t, cls in DocumentTag.tag_types.items():
            if len(soup.find("TAGS").findall(t)):
                for element in soup.find("TAGS").findall(t):
                    if "start" in element.attrib.keys() or \
                       "end" in element.attrib.keys():
                        self.tags.append(cls(element))
                    else:
                        self.doc_tags.append(DocumentTag(element))




class Evaluate(object):
    def __init__(self, annotator_cas, gold_cas, 
                 filters=None, conjunctive=False, invert=False):
        self.tp = []
        self.fp = []
        self.fn = []
        self.doc_ids = []
        self.invert = invert
        self.conjunctive = conjunctive
        self.verbose = False

        if filters == None:
            self.filters = []
        else:
            self.filters = filters


        assert len(set([a.sys_id for a in annotator_cas.values()])) == 1, \
            "More than one annotator ID in this set of Annotations!"
        

        self.sys_id = annotator_cas.values()[0].sys_id
                
        for doc_id in list(set(annotator_cas.keys()) & set(gold_cas.keys())):
            if filters != None:
            # Get all doc tags for each tag that passes all the 
            # predicate functions in filters
                if conjunctive:
                    if invert:
                        gold = set([t for t in self.get_tagset(gold_cas[doc_id]) 
                                    if not all( [f(t) for f in self.filters] ) ])
                        system = set([t for t in self.get_tagset(annotator_cas[doc_id]) 
                                      if not all( [f(t) for f in self.filters] )])
                    else:
                        gold = set([t for t in self.get_tagset(gold_cas[doc_id]) 
                                    if all( [f(t) for f in self.filters] ) ])
                        system = set([t for t in self.get_tagset(annotator_cas[doc_id]) 
                                      if all( [f(t) for f in self.filters] )])
                else:
                    if invert:
                        gold = set([t for t in self.get_tagset(gold_cas[doc_id]) 
                                    if not any( [f(t) for f in self.filters] ) ])
                        system = set([t for t in self.get_tagset(annotator_cas[doc_id]) 
                                      if not any( [f(t) for f in self.filters] )])
                    else:
                        gold = set([t for t in self.get_tagset(gold_cas[doc_id]) 
                                    if any( [f(t) for f in self.filters] ) ])
                        system = set([t for t in self.get_tagset(annotator_cas[doc_id]) 
                                      if any( [f(t) for f in self.filters] )])
                    
            else:
                gold = set(self.get_tagset(gold_cas[doc_id]))
                system = set(self.get_tagset(annotator_cas[doc_id]))

            self.tp.append(gold.intersection(system))
            self.fp.append(system - gold)
            self.fn.append(gold - system)
            self.doc_ids.append(doc_id)





    @staticmethod
    def recall(tp, fn):    
        try: 
            return len(tp) / float(len(fn) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def precision(tp, fp):
        try:
            return len(tp) / float(len(fp) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def F_beta(p, r, beta=1):
        try:
            return (1 + beta**2) * ((p * r) / (p + r))
        except ZeroDivisionError:
            return 0.0

    def macro_recall(self, tp, fn):
        np = numpy.array([Evaluate.recall(tp, fn) 
                          for tp,fn in zip(self.tp, self.fn)])
        return (np.mean(), np.std())

    def macro_precision(self, tp, fp):
        np = numpy.array([Evaluate.precision(tp, fp) 
                          for tp,fp in zip(self.tp, self.fp)])
        return (np.mean(), np.std())


    def micro_recall(self, tp, fn):
        try:
            return sum([len(t) for t in self.tp]) /  \
                float(sum([len(t) for t in self.tp]) +
                      sum([len(t) for t in self.fn]))
        except ZeroDivisionError:
            return 0.0

    def micro_precision(self, tp, fp):
        try:
            return sum([len(t) for t in self.tp]) /  \
                float(sum([len(t) for t in self.tp]) + 
                      sum([len(t) for t in self.fp]))
        except ZeroDivisionError:
            return 0.0



    def _print_docs(self):
        for i,doc_id in enumerate(self.doc_ids):
            mp = Evaluate.precision(self.tp[i], self.fp[i])
            mr = Evaluate.recall(self.tp[i], self.fn[i])
            str_fmt = "{:<25}{:<15}{:<15}{:<20}"

            print(str_fmt.format(doc_id, 
                                 "Precision", "", 
                                 "{:.4}".format(mp)))

            print(str_fmt.format("[{}({}){}]".format(len(self.tp[i]) + 
                                                     len(self.fn[i]), 
                                                     len(self.tp[i]),
                                                     len(self.tp[i]) + 
                                                     len(self.fp[i])),
                                 "Recall",    
                                 "", 
                                 "{:.4}".format(mr)))

            print(str_fmt.format("",     
                                 "F1",        
                                 "", 
                                 "{:.4}".format(Evaluate.F_beta(mp, mr))))

            print("{:-<25}{:-<15}{:-<15}{:-<20}".format("", "", "", ""))



    def _print_summary(self):
        Mp, Mp_std = self.macro_precision(self.tp, self.fp)
        Mr, Mr_std = self.macro_recall(self.tp, self.fn)
        mp = self.micro_precision(self.tp, self.fp)
        mr = self.micro_recall(self.tp, self.fn)

        str_fmt = "{:<25}{:<15}{:<15}{:<20}"

        print(str_fmt.format(self.sys_id + 
                             " ({})".format(len(self.doc_ids)), 
                             "Measure", "Macro (SD)", "Micro") )

        print("{:-<25}{:-<15}{:-<15}{:-<20}".format("", "", "", ""))

        print(str_fmt.format("Total", 
                             "Precision", 
                             "{:.4} ({:.2})".format(Mp, Mp_std),      
                             "{:.4}".format(mp)))

        print(str_fmt.format("",      
                             "Recall",    
                             "{:.4} ({:.2})".format(Mr, Mr_std),      
                             "{:.4}".format(mr)))

        print(str_fmt.format("",      
                             "F1",        
                             "{:.4}".format(Evaluate.F_beta(Mp, Mr)), 
                             "{:.4}".format(Evaluate.F_beta(mr, mp))))
        print("\n")

    def print_docs(self):
        print("Report for {}:".format(self.sys_id))
        print("{:<25}{:<15}{:<15}{:<20}".format("", "Measure", "", "Micro") )
        print("{:-<25}{:-<15}{:-<15}{:-<20}".format("", "", "", ""))
        self._print_docs()


    def print_report(self, verbose=False):
        self.verbose = verbose
        if verbose:
            self.print_docs()

        self._print_summary()


    def get_tagset(self, annotation):
        raise Exception("Must be implemented by Subclass!")



class EvaluatePHI(Evaluate):
    def get_tagset(self, annotation):
        return annotation.get_phi()



class EvaluateCardiacRisk(Evaluate):
    def get_tagset(self, annotation):
        return annotation.get_doc_tags() 




class CombinedEvaluation(object):
    """Base class for running multiple evaluations. This has a similar function
    signature to Evaluate and so can be used interchangably in the evaluate() 
    function.
    """
    def __init__(self):
        self.evaluations = []

    def add_eval(self, e, label=""):
        e.sys_id = e.sys_id + " " + label if e.sys_id and e.sys_id != '' else label
        self.evaluations.append(e)
        
    def print_docs(self):
        for e in self.evaluations:
            e.print_docs()
        
    def print_report(self, verbose=False):
        for e in self.evaluations:
            e.print_report(verbose=verbose)



class CardiacRiskTrackEvaluation(CombinedEvaluation):
    
    def __init__(self, annotator_cas, gold_cas, **kwargs):

        super(CardiacRiskTrackEvaluation, self).__init__()
        # Basic Evaluation
        self.add_eval(EvaluateCardiacRisk(annotator_cas, gold_cas, **kwargs), label="")


class PHITrackEvaluation(CombinedEvaluation):
    
    # list of Tuples of regular expressions for matching (TAG, TYPE)
    # That are considered to be HIPAA protected for the PHI Track evaluation
    HIPAA_regexes = [(re.compile("NAME"), re.compile("PATIENT")),
                     (re.compile("LOCATION"), re.compile("CITY")),
                     (re.compile("LOCATION"), re.compile("STREET")),
                     (re.compile("LOCATION"), re.compile("ZIP")),
                     (re.compile("LOCATION"), re.compile("ORGANIZATION")),
                     (re.compile("DATE"), re.compile(".*")),
                     (re.compile("CONTACT"), re.compile("PHONE")),
                     (re.compile("CONTACT"), re.compile("FAX")),
                     (re.compile("CONTACT"), re.compile("EMAIL")),
                     (re.compile("ID"), re.compile("SSN")),
                     (re.compile("ID"), re.compile("MEDICALRECORD")),
                     (re.compile("ID"), re.compile("HEALTHPLAN")),
                     (re.compile("ID"), re.compile("ACCOUNT")),
                     (re.compile("ID"), re.compile("LICENSE")),
                     (re.compile("ID"), re.compile("VEHICLE")),
                     (re.compile("ID"), re.compile("DEVICE")),
                     (re.compile("ID"), re.compile("BIOID")),
                     (re.compile("ID"), re.compile("IDNUM ")),
                     (re.compile("AGE"), re.compile(".*"))]



    def __init__(self, annotator_cas, gold_cas, **kwargs):

        super(PHITrackEvaluation, self).__init__()

        # Basic Evaluation
        self.add_eval(EvaluatePHI(annotator_cas, gold_cas, **kwargs), label="Strict")

        # Fuzzy Evaluation
        PHITag.fuzzy_end_equality(2)
        self.add_eval(EvaluatePHI(annotator_cas, gold_cas, **kwargs), label="Relaxed")


        # Add HIPAA filter to evaluation arguments
        kwargs['filters'] = [PHITrackEvaluation.HIPAA_predicate_filter]

        # Change equality back to strict
        PHITag.strict_equality()
        self.add_eval(EvaluatePHI(annotator_cas, gold_cas, **kwargs), 
                      label="HIPAA Strict")

        # Change equality to fuzzy end
        PHITag.fuzzy_end_equality(2)
        self.add_eval(EvaluatePHI(annotator_cas, gold_cas, **kwargs), 
                      label="HIPAA Relaxed")
        
                
    @staticmethod
    def HIPAA_predicate_filter(tag):
        return any([n_re.match(tag.name) and t_re.match(tag.TYPE) 
                for n_re, t_re in PHITrackEvaluation.HIPAA_regexes])
