from pathlib import Path
import glob,shutil,json,subprocess,imageio_ffmpeg,imageio.v2 as iio
out=Path('artifacts/class_progress_2026_10_08/comparison_demos');src={}
for name,folder in [('safelibero_without','baseline_og10_L1_video'),('safelibero_with','fol_v19_spatial_L1_n50_video')]:
 p=glob.glob(f'.worktrees/fol-safety-filter/{folder}/safelibero_spatial/0/I/*.mp4')[0];shutil.copyfile(p,out/(name+'.mp4'));src[name]={'path':p,'playback_multiplier':1}
for name,arm,suffix in [('liberosafety_without','base','fail'),('liberosafety_with','nogt','succ')]:
 p=f'.worktrees/flow-guidance-tier-gt/ls_teaser_render/{arm}_oah_L2/obstacle_avoidance_human/L2/videos/t14_ep1_{suffix}.mp4'
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',p,'-vf','setpts=0.5*PTS','-r','12','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/(name+'.mp4'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 src[name]={'path':p,'playback_multiplier':2}
for name in src:
 r=iio.get_reader(out/(name+'.mp4'));m=r.get_meta_data();src[name].update(duration=m['duration'],size=m['size']);r.get_data(0);r.get_data(max(0,int(m['duration']*m['fps'])-2));r.close()
json.dump(src,open(out/'sources.json','w'),indent=2);print(json.dumps(src,indent=2))
