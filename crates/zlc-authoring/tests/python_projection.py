"""The Python facade uses the same projector as native workbench fields."""
import pathlib
import sys
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[3]/'science'/'src'))
from zlc_atom.authoring import AuthoringField, AuthoringSchema, AuthoringChoice, project_field

class Projection(unittest.TestCase):
    def test_native_types_choices_ranges_and_required_fields(self):
        fields=AuthoringSchema((AuthoringField('i','int','Integer',0),AuthoringField('n','float','Number',1.0,minimum=0),
            AuthoringField('b','bool','Flag',False),AuthoringField('p','pair','Pair',[1,2])))
        self.assertEqual(fields.project_values({'i':'+123','n':'2.5','b':'on','p':'3, 4'}),{'i':123,'n':2.5,'b':True,'p':[3,4]})
        for field,value in [('i',True),('i',1.1),('n',False),('b',1)]:
            with self.subTest(field=field),self.assertRaises(TypeError): fields.project_values({field:value})
        with self.assertRaises(ValueError): fields.project_values({'n':-1})
        required=AuthoringSchema((AuthoringField('s','str','Text','',required=True),))
        self.assertEqual(required.draft_values({}),{'s':''})
        with self.assertRaises(ValueError): required.project_values({})

    def test_disabled_draft_banks_are_preserved_even_when_not_projectable(self):
        columns=(AuthoringField('on','bool','On',False),
                 AuthoringField('values','numeric_tuple','Values',(1,),enabled_when=('on',(True,))))
        schema=AuthoringSchema((AuthoringField('rows','rows','Rows',(),columns=columns),))
        dormant={'not':'a numeric sequence'}
        self.assertEqual(schema.project_values({'rows':[{'on':False,'values':dormant}]}),{'rows':({'on':False,'values':dormant},)})
        with self.assertRaises(TypeError): schema.project_values({'rows':[{'on':True,'values':dormant}]})

    def test_science_validator_runs_only_for_complete_projected_drafts(self):
        seen=[]
        schema=AuthoringSchema((AuthoringField('text','str','Text','',required=True),),validator=lambda values:seen.append(values.copy()))
        schema.draft_values({})
        self.assertEqual(seen,[])
        schema.project_values({'text':'ready'})
        self.assertEqual(seen,[{'text':'ready'}])

if __name__=='__main__':unittest.main()
