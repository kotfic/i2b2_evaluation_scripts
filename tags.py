from lxml import etree
from collections import OrderedDict

###############
##    Class hierarchy reference
##
##    [+]Tag
##       [+]DocumentTag
##       [+]AnnotatorTag
##          [+]PHITag
##             [+]NameTag       
##             [+]ProfessionTag 
##             [+]LocationTag   
##             [+]AgeTag        
##             [+]DateTag       
##             [+]ContactTag    
##             [+]IDTag         
##             [+]OtherTag      
##          [+]DiseaseTag
##             [+]DiabetesTag
##             [+]CADTag
##             [+]HypertensionTag
##             [+]HyperlipidemiaTag
##             [+]ObeseTag
##             [+]MedicationTag
##          [+]FamilyHistTag
##          [+]SmokerTag


class MalformedTagException(Exception):
    pass

class InvalidAttributException(Exception):
    pass

def isint(value):
    try:
        return int(value)
    except:
        return False



class Tag(object):
    attributes = OrderedDict()

    def __init__(self, element):
        self.name = element.tag
        try:
            self.id = element.attrib['id']
        except KeyError:
            self.id = ""

    def _get_key(self):
        key = []
        for k in self.key:
            key.append(getattr(self, k))
        return tuple(key)


    def __eq__(self, other):
        return self._get_key() == other._get_key() and other._get_key() == self._get_key()

    def __hash__(self):
        return hash(self._get_key())

    def is_valid(self):
        for k, validp in self.attributes.items():
            try:
                # If the validating function fails throw false
                if not validp(getattr(self,k)):
                    return False
            except AttributeError:
                # Attribute is not set,  if it is in the key then
                # it is a required attribute and we return false.
                if k in self.key:
                    return False

        return True

    def toElement(self):
        element = etree.Element(self.name)
        for k, validp in self.attributes.items():
            try:
                if validp(getattr(self,k)):
                    element.attrib[k] = getattr(self, k)
                else:
                    element.attrib[k] = getattr(self, k)
                    print("WARNING: Expected attribute '%s' for tag %s was not valid ('%s')" % (k, "<%s (%s)>" % (self.name, self.id), getattr(self, k)))
            except AttributeError:
                if k in self.key:
                    element.attrib[k] = ''
                    print("WARNING: Expected attribute '%s' for tag %s" % (k, "<%s, %s>" % (self.name, self.id)))

        return element

    def __repr__(self):
        return "<{0}: {1}>".format(self.__class__.__name__, ", ".join(self._get_key()))


    def toXML(self):
        return etree.tostring(self.toElement())


    def toDict(self, attributes=None):
        d = {}
        if attributes == None:
            attributes = ["name"] + [k for k, v in self.attributes.items()]

        for a in attributes:
            try:
                d[a] = getattr(self, a)
            except AttributeError:
                d[a] = None

        return d



class AnnotatorTag(Tag):
    attributes = OrderedDict()
    attributes["id"] = lambda v: True
    attributes["docid"] = lambda v: True
    attributes["start"] = isint
    attributes["end"] = isint
    attributes["text"] = lambda v: True


    key = ["name"]


    def __repr__(self):
        try:
            return "<{0}: {1} s:{2} e:{3}>".format(self.__class__.__name__, ", ".join(self._get_key()), self.start, self.end )
        except AttributeError:
            return super(Tag, self).__repr__()


    def __init__(self, element):
        super(AnnotatorTag, self).__init__(element)
        self.id = None


        for k,validp in self.attributes.items():
            if k in element.attrib.keys():
                if validp(element.attrib[k]):
                    setattr(self, k, element.attrib[k])
                else:
                    
                    print("WARNING: Expected attribute '%s' for xml element <%s (%s)>  was not valid ('%s')" % (k, element.tag, element.attrib['id'], element.attrib[k]) )
                    setattr(self, k, element.attrib[k])
            elif k in self.key:
                print("WARNING: Expected attribute '%s' for xml element <%s ('%s')>, setting to ''" % (k, element.tag, element.attrib['id']) )
                setattr(self, k, '')

    def validate(self):
        for k,validp in self.attributes.items():
            try:
                if validp(getattr(self, k)):
                    continue
                else:
                    return False
            except AttributeError:
                if k in self.key:
                    return False

        return True
        


    def get_document_annotation(self):
        element = etree.Element(self.name)
        for k,v in zip(self.key, self._get_key()):
            element.attrib[k] = v
        return DocumentTag(element)


    def get_start(self):
        try:
            return int(self.start)
        except TypeError:
            return self.start

    def get_end(self):
        try:
            return int(self.end)
        except TypeError:
            return self.end


class PHITag(AnnotatorTag):
    valid_TYPE = "PATIENT", "DOCTOR", "USERNAME", "PROFESSION", "ROOM", "DEPARTMENT", "HOSPITAL", "ORGANIZATION", "STREET", "CITY", "STATE", "COUNTRY", "ZIP", "OTHER", "LOCATION-OTHER", "AGE", "DATE", "PHONE", "FAX", "EMAIL", "URL", "IPADDR", "SSN", "MEDICALRECORD", "HEALTHPLAN", "ACCOUNT", "LICENSE", "VEHICLE", "DEVICE", "BIOID", "IDNUM"
    attributes = OrderedDict(AnnotatorTag.attributes.items())
    attributes['TYPE'] = lambda v: v in PHITag.valid_TYPE

    key = AnnotatorTag.key + ["start", "end", "TYPE"]


class NameTag(PHITag):
    valid_TYPE = [ 'PATIENT','DOCTOR','USERNAME' ]
    attributes = OrderedDict(PHITag.attributes.items())
    attributes['TYPE'] = lambda v: v in NameTag.valid_TYPE


class ProfessionTag(PHITag):
    valid_TYPE = ["PROFESSION"]
    attributes = OrderedDict(PHITag.attributes.items())
    attributes['TYPE'] = lambda v: v in ProfessionTag.valid_TYPE

class LocationTag(PHITag):
    valid_TYPE = ['ROOM','DEPARTMENT','HOSPITAL','ORGANIZATION','STREET','CITY','STATE','COUNTRY','ZIP','LOCATION-OTHER']
    attributes = OrderedDict(PHITag.attributes.items())
    attributes['TYPE'] = lambda v: v in LocationTag.valid_TYPE

class AgeTag(PHITag):
    valid_TYPE = ['AGE']
    attributes = OrderedDict(PHITag.attributes.items())
    attributes['TYPE'] = lambda v: v in AgeTag.valid_TYPE

class DateTag(PHITag):
    valid_TYPE = ['DATE']
    attributes = OrderedDict(PHITag.attributes.items())
    attributes['TYPE'] = lambda v: v in DateTag.valid_TYPE

class ContactTag(PHITag):
    valid_TYPE = ['PHONE','FAX','EMAIL','URL','IPADDR']
    attributes = OrderedDict(PHITag.attributes.items())
    attributes['TYPE'] = lambda v: v in ContactTag.valid_TYPE

class IDTag(PHITag):
    valid_TYPE = ['SSN','MEDICALRECORD','HEALTHPLAN','ACCOUNT','LICENSE','VEHICLE','DEVICE','BIOID','IDNUM']
    attributes = OrderedDict(PHITag.attributes.items())
    attributes['TYPE'] = lambda v: v in IDTag.valid_TYPE

class OtherTag(PHITag):
    valid_TYPE =  'OTHER'
    attributes = OrderedDict(PHITag.attributes.items())
    attributes['TYPE'] = lambda v: v in OtherTag.valid_TYPE
        


PHITag.tag_types = {
    "PHI" : PHITag,
    "NAME" : NameTag,
    "PROFESSION" : ProfessionTag,
    "LOCATION" : LocationTag,
    "AGE" : AgeTag,
    "DATE" : DateTag,
    "CONTACT" : ContactTag,
    "ID" : IDTag,
    "OTHER" : OtherTag }




class FamilyHistTag(AnnotatorTag):
    valid_indicator = ["present", "not present"]

    attributes = OrderedDict(AnnotatorTag.attributes.items())
    attributes['indicator'] = lambda v: v in FamilyHistTag.valid_indicator

    key = AnnotatorTag.key + ["indicator"]

    def __init__(self, element):
        # FAMILY_HIST tags do not (by design) have an indicator tag before cleaning
        # However we require an indicator to be a valid tag after cleaning,  this causes
        # some unfortunate hacking here to ensure that an indicator tag is present before
        # any validation happens.
        try:
            if int(element.attrib['start'] ) != -1 and int(element.attrib['end']) != -1:
                element.attrib['indicator'] = "present"
            else:
                element.attrib['indicator'] = "not present"
        except (AttributeError, KeyError):
            element.attrib['indicator'] = "not present"

        super(FamilyHistTag, self).__init__(element)


class SmokerTag(AnnotatorTag):
    valid_status = ["current", "past", "ever", "never", "unknown" ]

    attributes = OrderedDict(AnnotatorTag.attributes.items())
    attributes["status"] = lambda v: v in SmokerTag.valid_status

    key = AnnotatorTag.key + ["status"]



class DiseaseTag(AnnotatorTag):
    valid_time = [ "before DCT", "during DCT", "after DCT", "continuing" ]

    attributes = OrderedDict(AnnotatorTag.attributes.items())
    attributes["time"] = lambda v: v in DiseaseTag.valid_time

    key = AnnotatorTag.key + ["time"]



class DiabetesTag(DiseaseTag):
    valid_indicator = [ "mention", "A1C", "glucose" ]

    attributes = OrderedDict(DiseaseTag.attributes.items())
    attributes["indicator"] = lambda v: v in DiabetesTag.valid_indicator

    key = DiseaseTag.key + ["indicator"]



class CADTag(DiseaseTag):
    valid_indicator = [ "mention", "event", "test", "symptom" ]

    attributes = OrderedDict(DiseaseTag.attributes.items())
    attributes["indicator"] = lambda v: v in CADTag.valid_indicator

    key = DiseaseTag.key + ["indicator"]




class HypertensionTag(DiseaseTag):
    valid_indicator = [ "mention", "high bp"]

    attributes = OrderedDict(DiseaseTag.attributes.items())
    attributes["indicator"] = lambda v: v in HypertensionTag.valid_indicator

    key = DiseaseTag.key + ["indicator"]



class HyperlipidemiaTag(DiseaseTag):
    valid_indicator = [ "mention",  "high chol.",  "high LDL" ]

    attributes = OrderedDict(DiseaseTag.attributes.items())
    attributes["indicator"] = lambda v: v in HyperlipidemiaTag.valid_indicator

    key = DiseaseTag.key + ["indicator"]




class ObeseTag(DiseaseTag):
    valid_indicator = ["mention", "BMI", "waist circum."]

    attributes = OrderedDict(DiseaseTag.attributes.items())
    attributes["indicator"] = lambda v: v in ObeseTag.valid_indicator

    key = DiseaseTag.key + ["indicator"]



class MedicationTag(DiseaseTag):
    valid_type1 = [ "ace inhibitor", "amylin", "anti diabetes", "arb", "aspirin", "beta blocker", "calcium channel blocker", "diuretic", "dpp4 inhibitors", "ezetimibe", "fibrate", "insulin", "metformin", "niacin", "nitrate", "statin", "sulfonylureas", "thiazolidinedione", "thienopyridine" ]
    valid_type2 = [ "ace inhibitor", "amylin", "anti diabetes", "arb", "aspirin", "beta blocker", "calcium channel blocker", "diuretic", "dpp4 inhibitors", "ezetimibe", "fibrate", "insulin", "metformin", "niacin", "nitrate", "statin", "sulfonylureas", "thiazolidinedione", "thienopyridine", "" ]

    attributes = OrderedDict(DiseaseTag.attributes.items())
    attributes["type1"] = lambda v: v.lower() in MedicationTag.valid_type1
    attributes["type2"] = lambda v: v.lower() in MedicationTag.valid_type2

    key = DiseaseTag.key + ["type1", "type2"]




class DocumentTag(Tag):
    tag_types = {"DIABETES" : DiabetesTag,
                 "CAD" : CADTag,
                 "HYPERTENSION" : HypertensionTag,
                 "HYPERLIPIDEMIA" : HyperlipidemiaTag,
                 "SMOKER" : SmokerTag,
                 "OBESE" : ObeseTag,
                 "FAMILY_HIST" : FamilyHistTag,
                 "MEDICATION" : MedicationTag }


    def __init__(self, element):
        super(DocumentTag, self).__init__(element)

        self.key = self.tag_types[self.name].key
        self.attributes = self.tag_types[self.name].attributes

        self.annotator_tags = []

        for k in self.key:
            try:
                setattr(self, k, element.attrib[k])
            except KeyError:
                continue

        for e in element:
            cls = self.tag_types[e.tag]
            self.annotator_tags.append(cls(e))

    def toTagType(self):
        element = super(DocumentTag, self).toElement()
        cls = self.tag_types[self.name]

        return cls(element)

    def toElement(self, with_annotator_tags=True):
        element = super(DocumentTag, self).toElement()
        if with_annotator_tags:
            for at in self.annotator_tags:
                element.append(at.toElement())

        return element


PHI_TAG_CLASSES = [NameTag,
                   ProfessionTag,
                   LocationTag,   
                   AgeTag,  
                   DateTag,       
                   ContactTag,
                   IDTag,   
                   OtherTag]

# Comment should be last in tag order,  so add it down here
# that way all other sub tags have had their attributes set first
# This also provides the MEDICAL_TAG_CLASSES list.
MEDICAL_TAG_CLASSES = [FamilyHistTag,
                       SmokerTag,
                       DiseaseTag,
                       DiabetesTag,
                       CADTag,
                       HypertensionTag,
                       HyperlipidemiaTag,
                       ObeseTag,
                       MedicationTag]

for c in MEDICAL_TAG_CLASSES + PHI_TAG_CLASSES:
    c.attributes["comment"] = lambda v: True
