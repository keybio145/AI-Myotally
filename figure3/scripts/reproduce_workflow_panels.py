"""Redraw Figure 3 stages from supplied endpoint masks and saved filtering decisions."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import workflow_display as w
ROOT=Path(__file__).resolve().parents[2]
MEASUREMENTS=ROOT/'pca/data/tables/Fiber_Measurements_image_level_calibration.csv'
def load_data(image):
    states=ROOT/'data/editor_states';statefile=next(states.glob(image+'_*_state.json'));base=statefile.name[:-len('_state.json')]
    state=json.loads(statefile.read_text());final=np.asarray(Image.open(ROOT/f'data/masks/{image}_final_labels.tif')).astype(np.int32)
    reference=np.load(states/(base+'_filter_reference.npy'),allow_pickle=False)
    with np.load(states/(base+'_filter_destinations.npz'),allow_pickle=False) as z:
        raw=z['raw_segmentation'].astype(np.int32);non=z['non_myofiber'].astype(np.int32);geometry=z['geometry_excluded'].astype(np.int32)
    with Image.open(ROOT/f'data/images/{image}/Laminin.tif') as im:lam=w.safe_uint8_gray(np.asarray(im))
    assert final.shape==reference.shape==lam.shape
    t=pd.read_csv(MEASUREMENTS);t=t[t.Image==image]
    return {'state':state,'raw':raw,'reference':reference,'final':final,'non_myofiber':non,'geometry':geometry,'reject_preview':np.load(states/(base+'_pre_filter_rejects.npy'),allow_pickle=False),'laminin_gray':lam,'id_to_type':dict(zip(t.Mask_ID.astype(int),t.Fiber_Type)),'manual_filter_labels':w.manual_labels_from_state(state),'rf_statuses':w.rf_statuses_from_state(state)}
def scale_bar(image,um_per_pixel,length_um):
    draw=ImageDraw.Draw(image);bar=round(length_um/um_per_pixel);margin=20;x=image.width-margin-bar;y=image.height-margin
    draw.line((x,y,x+bar,y),fill='black',width=7);draw.line((x,y,x+bar,y),fill='white',width=3)
    try:font=ImageFont.truetype('arial.ttf',18)
    except OSError:font=ImageFont.load_default()
    draw.text((x,y-25),str(length_um)+' µm',fill='white',stroke_width=1,stroke_fill='black',font=font)
def render(image,output,display_boxes=None):
    data=load_data(image);stages=w.build_stage_images(data)
    box,detail=w.choose_roi(data['raw'],data['reference'],data['final'],stages['event_mask'],(1500,900))
    label='Old 3' if image=='O3' else 'Young 1'
    out=Path(output)/label.replace(' ','_');out.mkdir(parents=True,exist_ok=False)
    info={'sample':label,'stages':{},'scope':'Saved mask/state rendering only; no segmentation, RF training or typing recalibration; no final PDF byte-reconstruction claim'}
    for panel in 'ABCDE':
        rgb=stages[panel];im=Image.fromarray(rgb)
        selected=(display_boxes or {}).get(image,{}).get(panel,box)
        x,y,width,height=map(int,selected)
        zoom=im.crop((x,y,x+width,y+height));scale_bar(zoom,.36,100);zoom.save(out/f'Figure3{panel}_zoom.png')
        preview=im.copy();preview.thumbnail((1500,1100));sx=preview.width/im.width;sy=preview.height/im.height
        draw=ImageDraw.Draw(preview);draw.rectangle((round(x*sx),round(y*sy),round((x+width)*sx),round((y+height)*sy)),outline='white',width=2)
        scale_bar(preview,.36/sx,500);preview.save(out/f'Figure3{panel}_whole.png')
        info['stages'][panel]={'display_box_xywh':[x,y,width,height],'source_shape':list(rgb.shape),'canonical_rgb_sha256':hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),'preview':f'Figure3{panel}_whole.png','zoom':f'Figure3{panel}_zoom.png'}
    (out/'display_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    return info
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--sample',choices=['O3','Y1','both'],default='both');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    boxes=ROOT/'figure3/display_boxes.json';boxes=json.loads(boxes.read_text()) if boxes.exists() else None
    reports=[render(im,a.output,boxes) for im in (['O3','Y1'] if a.sample=='both' else [a.sample])]
    (a.output/'Figure3_display_report.json').write_text(json.dumps(reports,indent=2)+'\n');print('Figure 3 saved-state panels reproduced')
if __name__=='__main__':main()
