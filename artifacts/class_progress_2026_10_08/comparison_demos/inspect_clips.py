from pathlib import Path
import glob,imageio.v2 as iio
from PIL import Image,ImageDraw
out=Path('artifacts/class_progress_2026_10_08/comparison_demos')
files=[]
for task in range(4):
 for conf in ['baseline_og10_L1_video','fol_v20_spatial_L1_n50_video']:
  files+=glob.glob(f'.worktrees/fol-safety-filter/{conf}/safelibero_spatial/{task}/I/*.mp4')
files+=glob.glob('.worktrees/flow-guidance-tier-gt/ls_teaser_render/*oah*/*/*/videos/t14_ep1_*.mp4')
for start in range(0,len(files),4):
 sheet=Image.new('RGB',(1000,280*min(4,len(files)-start)),'white');dr=ImageDraw.Draw(sheet)
 for k,p in enumerate(files[start:start+4]):
  r=iio.get_reader(p);m=r.get_meta_data();dur=m['duration'];dr.text((5,k*280+3),p.split('/')[2]+' / '+Path(p).name[:67],fill='black');print(p,m['size'],dur,m['fps'])
  for j,t in enumerate([0,dur/3,2*dur/3,max(0,dur-.15)]):
   im=Image.fromarray(r.get_data(int(t*m['fps'])));im.thumbnail((250,250));sheet.paste(im,(j*250,k*280+25))
  r.close()
 sheet.save(out/f'options_{start//4}.png')
