"""Verify and reproduce Figure 3, Figure 4, Result 4 and descriptive PCA from supplied materials."""
import argparse,csv,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parent
TABLE=ROOT/'pca/data/tables/Fiber_Measurements_image_level_calibration.csv'
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def compare_csv(a,b):
    actual=pd.read_csv(a,keep_default_na=False);expected=pd.read_csv(b,keep_default_na=False)
    if actual.shape!=expected.shape or list(actual)!=list(expected):raise AssertionError('CSV shape/columns differ: '+str(b))
    for c in expected:
        if actual[c].astype(str).equals(expected[c].astype(str)):continue
        av=pd.to_numeric(actual[c],errors='coerce').to_numpy(float);ev=pd.to_numeric(expected[c],errors='coerce').to_numpy(float)
        if not np.isfinite(av).all() or not np.isfinite(ev).all():raise AssertionError('Text/category differs: '+c)
        np.testing.assert_allclose(av,ev,rtol=1e-7,atol=1e-8)
    return {'file':Path(b).name,'rows':len(actual),'passed':True}
def verify_inputs():
    count=0
    for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
        digest,relative=line.split('  ',1);p=(ROOT/relative).resolve()
        if not p.is_relative_to(ROOT):raise ValueError(relative)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise AssertionError('SHA256 mismatch: '+relative)
        count+=1
    binding=json.loads((ROOT/'metadata/measurement_restoration.json').read_text());fields=binding['original_columns']
    with TABLE.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    actual=hashlib.sha256(json.dumps([[r[c] for c in fields] for r in rows],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if actual!=binding['original_parsed_values_sha256']:raise AssertionError('Original 27 fields changed')
    t=pd.read_csv(TABLE);masks=[]
    for im in ['Y1','Y2','Y3','O1','O2','O3']:
        a=np.asarray(Image.open(ROOT/f'data/masks/{im}_final_labels.tif'));s=t[t.Image==im]
        assert set(map(int,np.unique(a)))-{0}==set(s.Mask_ID.astype(int))
        assert s.Fiber_ID.is_unique and s.Mask_ID.is_unique
        masks.append({'Image':im,'fibers':len(s),'labels_match':True})
    return {'manifest_entries':count,'original_27_fields_exactly_preserved':True,'final_mask_mapping':masks}
def figure4(out,render=False):
    import matplotlib.pyplot as plt
    out.mkdir();src=out/'SourceData';src.mkdir()
    g=module(ROOT/'figure4/scripts/figure4_numerical.py','f4numeric');g.SOURCE_DIR=src
    f=pd.read_csv(TABLE);f['Group']=f.Image.str[0].map(g.GROUP_MAP)
    for c,values in [('Image',g.SAMPLE_ORDER),('Group',g.GROUP_ORDER),('Fiber_Type',g.TYPE_ORDER)]:f[c]=pd.Categorical(f[c],values,ordered=True)
    g.panel_b_ecdf(f);g.morphometry_tables(f)
    rows=[]
    for im in g.SAMPLE_ORDER:
        s=f[f.Image==im]
        for t in g.TYPE_ORDER:rows.append({'Image':im,'Group':g.GROUP_MAP[im[0]],'Fiber_Type':t,'Fiber_Count':int((s.Fiber_Type==t).sum()),'Proportion_percent':100*(s.Fiber_Type==t).sum()/len(s),'Total_Fibers':len(s)})
    a=pd.DataFrame(rows);stats=a.groupby(['Group','Fiber_Type']).Proportion_percent.agg(Group_Mean_percent='mean',Group_SD_percent='std').reset_index();a=a.merge(stats,on=['Group','Fiber_Type']);a['Panel']='A_candidate_sampleStacked'
    name='Figure4A_composition_source_data.csv';template=pd.read_csv(ROOT/'figure4/expected'/name)
    a=a.set_index(['Image','Fiber_Type']).loc[pd.MultiIndex.from_frame(template[['Image','Fiber_Type']])].reset_index();a=a[list(template)];a.to_csv(src/name,index=False)
    animal=[]
    for facet,types in [('All',g.TYPE_ORDER)]+[(t,[t]) for t in g.TYPE_ORDER[1:]]:
        for im in g.SAMPLE_ORDER:
            v=f[(f.Image==im)&f.Fiber_Type.isin(types)];animal.append({'Facet':facet,'Image':im,'Group':g.GROUP_MAP[im[0]],'n':len(v),'CSA_median_um2':float(v.CSA_um2.median()),'MinFeret_median_um':float(v.MinFeret_um.median())})
    animal=pd.DataFrame(animal);animal.to_csv(src/'Figure4C_animal_medians_white_circles.csv',index=False)
    animal.groupby(['Facet','Group'])[['CSA_median_um2','MinFeret_median_um']].mean().reset_index().to_csv(src/'morphometry_animal_equal.csv',index=False)
    a.groupby(['Group','Fiber_Type']).Proportion_percent.mean().reset_index().to_csv(src/'composition_animal_equal.csv',index=False)
    names=['Figure4A_composition_source_data.csv','Figure4B_ECDF_source_data.csv','Figure4C_morphometry_by_animal.csv','Figure4C_morphometry_by_animal_and_type.csv','Figure4C_animal_medians_white_circles.csv','composition_animal_equal.csv','morphometry_animal_equal.csv']
    checks=[compare_csv(src/n,ROOT/'figure4/expected'/n) for n in names]
    # Verify all original fiber fields shared with F and map SourceData by one-to-one keys.
    for name in ['Figure4F_PCA_and_fiber_source_data.csv','Figure4D_CSA_map_source_data.csv','Figure4E_density_map_source_data.csv']:
        ref=pd.read_csv(ROOT/'figure4/expected'/name);joined=ref[['Image','Fiber_ID']].merge(pd.read_csv(TABLE),on=['Image','Fiber_ID'],validate='one_to_one')
        for c in sorted(set(joined)&set(ref)-{'Image','Fiber_ID'}):
            if pd.api.types.is_numeric_dtype(ref[c]):np.testing.assert_allclose(joined[c],ref[c],rtol=1e-7,atol=1e-8,equal_nan=True)
            else:assert joined[c].fillna('').astype(str).equals(ref[c].fillna('').astype(str)),(name,c)
        checks.append({'file':name,'rows':len(ref),'shared_measurement_fields_match':True})
    if render:
        r=module(ROOT/'figure4/scripts/figure4_renderer.py','f4renderer');r.SOURCE_DIR=ROOT/'figure4/expected';r.FIG_DIR=out;r.PANEL_DIR=out/'panels';r.OUT_PNG=out/'Figure4_panel_preview.png';r.OUT_PDF=out/'Figure4_panel_preview.pdf';r.OUT_SVG=out/'Figure4_panel_preview.svg';r.main();r.export_panels('ABCDEF');plt.close('all')
    return checks
def displayed_values(out,pca_output):
    t=pd.read_csv(TABLE);counts=t.groupby(['Image','Fiber_Type']).size().unstack(fill_value=0).reindex(index=['Y1','Y2','Y3','O1','O2','O3'],columns=['Type I','Type IIA','Type IIX','Type IIB'],fill_value=0)
    perc=counts.div(counts.sum(axis=1),axis=0)*100;records=[]
    def check(label,expected,actual):records.append({'target':label,'expected_display':expected,'reproduced_display':actual,'passed':expected==actual})
    for group,ims in [('Young',['Y1','Y2','Y3']),('Old',['O1','O2','O3'])]:
        for typ in ['Type IIA','Type IIX','Type IIB']:
            v=perc.loc[ims,typ];check(group+' '+typ+' mean',{'Young':{'Type IIB':'52.5','Type IIX':'38.7','Type IIA':'8.7'},'Old':{'Type IIB':'52.7','Type IIX':'32.8','Type IIA':'14.4'}}[group][typ],f'{v.mean():.1f}')
            if typ=='Type IIX':check(group+' IIX range',{'Young':'34.6–41.0','Old':'31.8–34.5'}[group],f'{v.min():.1f}–{v.max():.1f}')
    check('Old 1 IIA','24.2',f"{perc.loc['O1','Type IIA']:.1f}");check('Type I count','23',str(int(counts['Type I'].sum())))
    for im,n in [('Y1',2264),('Y2',2402),('Y3',2802),('O1',2737),('O2',2653),('O3',2515)]:check(im+' fibers',str(n),str(int(counts.loc[im].sum())))
    for panel,name,expected,digits in [('D','Figure4D_CSA_map_source_data.csv',('243','1045'),0),('E','Figure4E_density_map_source_data.csv',('0.10','0.25'),2)]:
        v=pd.read_csv(ROOT/'figure4/expected'/name)
        check(panel+' lower scale',expected[0],f'{v.Scale_Vmin.dropna().iloc[0]:.{digits}f}');check(panel+' upper scale',expected[1],f'{v.Scale_Vmax.dropna().iloc[0]:.{digits}f}')
    meta=json.loads((pca_output/'recomputed/pca_run_metadata.json').read_text());stats=json.loads((pca_output/'recomputed/interpretation_metrics.json').read_text())
    for tag,expected in [('pca_12features',['24.9','20.1']),('pca_marker_excluded_8features',['36.5','26.2'])]:
        a=stats['analyses'][tag]
        for i,pc in enumerate(['PC1','PC2']):check(tag+' '+pc+' variance',expected[i],f"{a['explained_variance_pct'][i]:.1f}")
        check(tag+' PC2 type eta2 percent',('33' if tag=='pca_12features' else '21'),f"{100*a['PCs']['PC2']['eta2']['type']:.0f}")
    c=pd.read_csv(pca_output/'recomputed/pca_12features_coefficients.csv');e=c[c.Feature=='Local_Type_Entropy'][['PC1_coefficient','PC2_coefficient']].to_numpy(float)
    records.append({'target':'absolute entropy coefficient','criterion':'<=0.11','reproduced_max_abs':float(np.abs(e).max()),'passed':bool((np.abs(e)<=.11).all())})
    check('PCA complete cases','15373',str(meta['cohort']['complete_case_rows']))
    perc.to_csv(out/'composition_percent_by_animal.csv');(out/'displayed_value_checks.json').write_text(json.dumps(records,indent=2)+'\n')
    return records
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--render-figure4',action='store_true');ap.add_argument('--skip-figure3',action='store_true');a=ap.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    report={'input_checks':verify_inputs(),'scope':'Supplied calibrated table and saved masks/decisions to figures/statistics; no full interactive replay'}
    env=os.environ.copy();env['PYTHONNOUSERSITE']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    subprocess.run([sys.executable,'-B',str(ROOT/'pca/jove_repro/scripts/verify_pca_reproduction.py'),'--output',str(a.output/'pca')],check=True,env=env)
    report['figure4']=figure4(a.output/'figure4',a.render_figure4);report['displayed_values']=displayed_values(a.output,a.output/'pca')
    if not a.skip_figure3:
        subprocess.run([sys.executable,'-B',str(ROOT/'figure3/scripts/reproduce_workflow_panels.py'),'--output',str(a.output/'figure3')],check=True,env=env)
        expected=json.loads((ROOT/'figure3/expected_stage_rgb_sha256.json').read_text())
        for image,label in [('O3','Old_3'),('Y1','Young_1')]:
            actual=json.loads((a.output/'figure3'/label/'display_manifest.json').read_text())
            for panel in 'ABCDE':assert actual['stages'][panel]['canonical_rgb_sha256']==expected[image][panel]
        report['figure3_saved_stage_rgb_checks']='10/10 matched; final PDF pixel identity not claimed' 
    report['all_numeric_checks_passed']=all(r['passed'] for r in report['displayed_values'])
    (a.output/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n');print('Numerical checks:',report['all_numeric_checks_passed']);raise SystemExit(0 if report['all_numeric_checks_passed'] else 2)
if __name__=='__main__':main()
