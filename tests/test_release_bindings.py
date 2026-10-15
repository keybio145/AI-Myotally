from pathlib import Path
import csv,hashlib,json,unittest
import numpy as np
import pandas as pd
from PIL import Image
R=Path(__file__).resolve().parents[1]
TABLE=R/'pca/data/tables/Fiber_Measurements_image_level_calibration.csv'
class ReleaseBindings(unittest.TestCase):
    def test_original_27_values_are_frozen(self):
        b=json.loads((R/'metadata/measurement_restoration.json').read_text());cols=b['original_columns']
        with TABLE.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        digest=hashlib.sha256(json.dumps([[r[c] for c in cols] for r in rows],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(digest,b['original_parsed_values_sha256']);self.assertEqual(len(rows),15373)
    def test_final_mask_id_mapping(self):
        t=pd.read_csv(TABLE)
        for image in ['Y1','Y2','Y3','O1','O2','O3']:
            a=np.asarray(Image.open(R/f'data/masks/{image}_final_labels.tif'));s=t[t.Image==image]
            self.assertEqual(set(map(int,np.unique(a)))-{0},set(s.Mask_ID.astype(int)))
            self.assertTrue(s.Fiber_ID.is_unique and s.Mask_ID.is_unique)
    def test_only_final_object_overrides_are_applied(self):
        t=pd.read_csv(TABLE);labels=pd.read_csv(R/'data/calibration/saved_typing_overrides.csv')
        expected=labels.merge(t[['Image','Mask_ID']],on=['Image','Mask_ID'],validate='one_to_one')
        actual=t[t.Typing_Review_Flag=='Manual override']
        self.assertEqual(set(zip(expected.Image,expected.Mask_ID)),set(zip(actual.Image,actual.Mask_ID)))
        self.assertTrue((actual.Typing_Confidence=='High').all())
        missing=set(zip(labels.Image,labels.Mask_ID))-set(zip(expected.Image,expected.Mask_ID))
        self.assertEqual(missing,{('Y1',785),('Y1',802)})
if __name__=='__main__':unittest.main()
