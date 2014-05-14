import re
from lxml import etree
import os
import numpy
from collections import defaultdict, OrderedDict
from tags import *
import codecs



class StandoffAnnotation():

    id_parser = re.compile(r'^(\d+)-(\d+)(.*)\.xml')
    
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
        self.post_normalized_sentences =  []
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
            with open(file_name) as handle:
                self.parse_text_and_tags(handle.read())
                self.file_name = file_name
                
    @property
    def id(self):
        return self.patient_id + "-" + self.record_id

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
            
            with open(path, "w") as h:
                h.writelines([re.sub("^\s+<", "<", l + "\n") 
                              for l in xml.split("\n")])

        else:
            with open(path, "w") as h:
                h.write(self.toXML(**kwargs))
        
        return True
    
    def __repr__(self):
        return "<StandoffAnnotation (%s) %s: tags:%s phi:%s>" \
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
        
        soup = etree.fromstring(self.raw.encode("utf-8"))
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
            str_fmt = "{:<15}{:<15}{:<15}{:<20}"

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

            print("{:-<15}{:-<15}{:-<15}{:-<20}".format("", "", "", ""))



    def _print_summary(self):
        Mp, Mp_std = self.macro_precision(self.tp, self.fp)
        Mr, Mr_std = self.macro_recall(self.tp, self.fn)
        mp = self.micro_precision(self.tp, self.fp)
        mr = self.micro_recall(self.tp, self.fn)

        str_fmt = "{:<15}{:<15}{:<15}{:<20}"

        print(str_fmt.format(self.sys_id + 
                             " ({})".format(len(self.doc_ids)), 
                             "Measure", "Macro (SD)", "Micro") )

        print("{:-<15}{:-<15}{:-<15}{:-<20}".format("", "", "", ""))

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
        print("{:<15}{:<15}{:<15}{:<20}".format("", "Measure", "", "Micro") )
        print("{:-<15}{:-<15}{:-<15}{:-<20}".format("", "", "", ""))
        self._print_docs()


    def print_report(self, verbose=False):
        self.verbose = verbose
        if verbose:
            self.print_docs()

        self._print_summary()




class EvaluatePHI(Evaluate):
    def __init__(self, annotator_cas, gold_cas, 
                 filters=None, conjunctive=False, invert=False):
        super(EvaluatePHI, self).__init__(annotator_cas, gold_cas, 
                                          filters=filters, 
                                          conjunctive=conjunctive, 
                                          invert=invert)

        assert len(set([a.sys_id for a in annotator_cas.values()])) == 1, \
            "More than one annotator ID in this set of Annotations!"
        
        self.sys_id = annotator_cas.values()[0].sys_id
        
        
        for doc_id in list(set(annotator_cas.keys()) & set(gold_cas.keys())):
            if filters != None:
            # Get all doc tags for each tag that passes all the 
            # predicate functions in filters
                if conjunctive:
                    if invert:
                        gold = set([t for t in gold_cas[doc_id].get_phi() 
                                    if not all( [f(t) for f in self.filters] ) ])
                        system = set([t for t in annotator_cas[doc_id].get_phi() 
                                      if not all( [f(t) for f in self.filters] )])
                    else:
                        gold = set([t for t in gold_cas[doc_id].get_phi() 
                                    if all( [f(t) for f in self.filters] ) ])
                        system = set([t for t in annotator_cas[doc_id].get_phi() 
                                      if all( [f(t) for f in self.filters] )])
                else:
                    if invert:
                        gold = set([t for t in gold_cas[doc_id].get_phi() 
                                    if not any( [f(t) for f in self.filters] ) ])
                        system = set([t for t in annotator_cas[doc_id].get_phi() 
                                      if not any( [f(t) for f in self.filters] )])
                    else:
                        gold = set([t for t in gold_cas[doc_id].get_phi() 
                                    if any( [f(t) for f in self.filters] ) ])
                        system = set([t for t in annotator_cas[doc_id].get_phi() 
                                      if any( [f(t) for f in self.filters] )])
                    
            else:
                gold = set(gold_cas[doc_id].get_phi())
                system = set(annotator_cas[doc_id].get_phi())

            self.tp.append(gold.intersection(system))
            self.fp.append(system - gold)
            self.fn.append(gold - system)
            self.doc_ids.append(doc_id)



class EvaluateCardiacRisk(Evaluate):
    def __init__(self, annotator_cas, gold_cas, 
                 filters=None, conjunctive=False, invert=False):
        super(EvaluateCardiacRisk, self).__init__(annotator_cas, gold_cas, 
                                                  filters=filters, 
                                                  conjunctive=conjunctive, 
                                                  invert=invert)
        
        assert len(set([a.sys_id for a in annotator_cas.values()])) == 1, \
            "More than one annotator ID in this set of Annotations!"

        self.sys_id = annotator_cas.values()[0].sys_id
        
        
        for doc_id in list(set(annotator_cas.keys()) & set(gold_cas.keys())):
            if filters != None:
            # Get all doc tags for each tag that passes all the 
            # predicate functions in filters
                if conjunctive:
                    if invert:
                        gold = set([t for t in gold_cas[doc_id].get_doc_tags() 
                                    if not all( [f(t) for f in self.filters] ) ])
                        system = set([t for t in annotator_cas[doc_id].get_doc_tags() 
                                      if not all( [f(t) for f in self.filters] )])
                    else:
                        gold = set([t for t in gold_cas[doc_id].get_doc_tags() 
                                    if all( [f(t) for f in self.filters] ) ])
                        system = set([t for t in annotator_cas[doc_id].get_doc_tags() 
                                      if all( [f(t) for f in self.filters] )])
                else:
                    if invert:
                        gold = set([t for t in gold_cas[doc_id].get_doc_tags() 
                                    if not any( [f(t) for f in self.filters] ) ])
                        system = set([t for t in annotator_cas[doc_id].get_doc_tags() 
                                      if not any( [f(t) for f in self.filters] )])
                    else:
                        gold = set([t for t in gold_cas[doc_id].get_doc_tags() 
                                    if any( [f(t) for f in self.filters] ) ])
                        system = set([t for t in annotator_cas[doc_id].get_doc_tags() 
                                      if any( [f(t) for f in self.filters] )])
                    
            else:
                gold = set(gold_cas[doc_id].get_doc_tags())
                system = set(annotator_cas[doc_id].get_doc_tags())

            self.tp.append(gold.intersection(system))
            self.fp.append(system - gold)
            self.fn.append(gold - system)
            self.doc_ids.append(doc_id)
