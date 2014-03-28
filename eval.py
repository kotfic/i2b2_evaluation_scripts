from classes import StandoffAnnotation, EvaluatePHI, EvaluateCardiacRisk
import argparse
import os
from collections import defaultdict

def get_document_dict_by_annotator_id(annotator_dirs):
    documents = defaultdict(lambda : defaultdict(int))
    
    for d in annotator_dirs:
        for fn in os.listdir(d):
            ca = StandoffAnnotation(d + fn)
            documents[ca.annotator_id][ca.id] = ca
            
    return documents



def evaluate(annotator_dirs, gs, verbose=False, filters=None, invert=False, conjunctive=False, phi=False):
    gold_cas = {}

    for fn in os.listdir(gs):
        ca = StandoffAnnotation(gs + fn)
        gold_cas[ca.id] = ca
    
    for annotator_id, annotator_cas in get_document_dict_by_annotator_id(annotator_dirs).items():
        if phi:
            e = EvaluatePHI(annotator_cas, gold_cas, filters=filters, invert=invert, conjunctive=conjunctive)
        else:
            e = EvaluateCardiacRisk(annotator_cas, gold_cas, filters=filters, invert=invert, conjunctive=conjunctive)

        e.print_report(verbose=verbose)

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

