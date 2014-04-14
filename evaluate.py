from classes import StandoffAnnotation, EvaluatePHI, EvaluateCardiacRisk
import argparse
import os
from collections import defaultdict
from tags import DocumentTag, MEDICAL_TAG_CLASSES

def get_predicate_function(arg):
    
    attrs = []

    # Get a list of valid attributes for this argument
    if arg in DocumentTag.tag_types.keys():
        attrs.append("name")
    else:
        tag_attributes = ["valid_type1", "valid_type2", "valid_indicator", "valid_status", "valid_time", "valid_type"]
        for cls in MEDICAL_TAG_CLASSES:
            for attr in tag_attributes:
                try:
                    if arg in getattr(cls, attr):
                        # add the attribute,  strip out the "valid_" prefix
                        attrs.append(attr.replace("valid_", ""))
                except AttributeError:
                    continue
        # Delete these so we don't end up carrying around references in our function
        try:
            del tag_attributes
            del cls
            del attr
        except NameError:
            pass

    attrs = list(set(attrs))

    if len(attrs) == 0:
        print("WARNING: could not find valid class attribute for \"{}\", skipping.".format(arg))
        return lambda t: True
    
    def matchp(t):
        for attr in attrs:
            if attr == "name" and t.name == arg:
                return True                            
            else:
                try:
                    if getattr(t, attr).lower() == arg.lower():
                        return True                    
                except (AttributeError, KeyError):
                    pass
        return False

    return matchp


def get_document_dict_by_annotator_id(annotator_dirs):
    documents = defaultdict(lambda : defaultdict(int))
    
    for d in annotator_dirs:
        for fn in os.listdir(d):
            ca = StandoffAnnotation(d + fn)
            documents[ca.annotator_id][ca.id] = ca
            
    return documents



def evaluate(annotator_dirs, gs, verbose=False, filters=None, invert=False, conjunctive=False, phi=False):
    gold_cas = {}

    if os.path.isfile(gs):
        gs = StandoffAnnotation(gs)
        sys = StandoffAnnotation(annotator_dirs[0])
        if phi:
            e = EvaluatePHI({sys.id : sys}, {gs.id : gs}, filters=filters, invert=invert, conjunctive=conjunctive)
        else:
            e = EvaluateCardiacRisk({sys.id : sys}, {gs.id : gs}, filters=filters, invert=invert, conjunctive=conjunctive)
            
        e.print_docs()
        
    elif os.path.isdir(gs):
        for fn in os.listdir(gs):
            ca = StandoffAnnotation(gs + fn)
            gold_cas[ca.id] = ca
    
        for annotator_id, annotator_cas in get_document_dict_by_annotator_id(annotator_dirs).items():
            if phi:
                e = EvaluatePHI(annotator_cas, gold_cas, filters=filters, invert=invert, conjunctive=conjunctive)
            else:
                e = EvaluateCardiacRisk(annotator_cas, gold_cas, filters=filters, invert=invert, conjunctive=conjunctive)

            e.print_report(verbose=verbose)
    else:
        Exception("Must pass file.xml file.xml  or directory/ directory/ on command line!")

    return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To Write")
    parser.add_argument('--filter', help="Filters to apply,  use with invert and conjunction")
    parser.add_argument('--conjunctive', help="if multiple filters are applied,  should these be combined with 'ands' or 'ors'", action="store_true")
    parser.add_argument('--invert', help="Invert the list in required,  match only tags that do not match values in the required list", action="store_true")    
    parser.add_argument('-v', '--verbose', help="list full document by document scores", action="store_true")
    parser.add_argument("from_dirs", help="directories to pull documents from", nargs="+")
    parser.add_argument("to_dir", help="directories to save documents to")
    
    args = parser.parse_args()
    if args.filter:            
        evaluate(args.from_dirs, args.to_dir, 
                 verbose=args.verbose,
                 invert=args.invert,
                 conjunctive=args.conjunctive,
                 filters=[get_predicate_function(a) for a in  args.filter.split(",")],
                 phi=True)
    else:
        evaluate(args.from_dirs, args.to_dir, 
                 verbose=args.verbose,
                 phi=True)

