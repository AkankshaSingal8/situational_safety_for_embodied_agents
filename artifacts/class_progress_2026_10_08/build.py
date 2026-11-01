from pathlib import Path
import json,glob,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Ellipse
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
import imageio.v2 as iio
ROOT=Path.cwd(); OUT=ROOT/'artifacts/class_progress_2026_10_08'; A=OUT/'assets'
BG='#F6F5F1'; INK='#172B3A'; TEAL='#007F79'; ORANGE='#C75B35'; GRAY='#627382'; BLUE='#355C9A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.facecolor':BG,'figure.facecolor':BG,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
provenance={}
def savefig(name):
 plt.savefig(A/(name+'.png'),dpi=180,bbox_inches='tight',facecolor=BG);plt.close()
def grouped(name,labels,series,colors,ylim=100,ylabel='Episodes (%)'):
 fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(len(labels));w=.7/len(series)
 for j,(lab,v) in enumerate(series.items()):
  b=ax.bar(x+(j-(len(series)-1)/2)*w,v,w*.92,label=lab,color=colors[j]);ax.bar_label(b,fmt='%.1f',padding=5,fontsize=12)
 ax.set_xticks(x,labels);ax.set_ylim(0,ylim);ax.set_ylabel(ylabel);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=len(series),frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);savefig(name)
def readresult(p):
 d=json.load(open(p));o=d.get('overall',d)
 return {'tsr':o.get('TSR',o.get('overall_TSR'))*100,'car':o.get('CAR',o.get('overall_CAR'))*100,'n':o['total_episodes']}
base={}
for model in ['openvla_oft','pi05','cosmos','fastwam']:
 for suite in ['spatial','object','goal','long']:
  for level in ['I','II']:
   paths=glob.glob(f'results/safelibero/policy_baselines/{model}/safelibero_{suite}/**/*.json',recursive=True)
   if model=='fastwam' and suite=='spatial' and level=='I':paths=glob.glob('results/safelibero/policy_baselines/fastwam_50ep/**/*.json',recursive=True)
   paths=[p for p in paths if f'level{level}-' in p or f'/{level}/' in p]
   candidates=[]
   for p in paths:
    try:
     r=readresult(p)
     if r['n']==200:candidates.append((p,r))
    except (KeyError,TypeError):pass
   if candidates:
    p,r=sorted(candidates)[-1];base[(model,suite,level)]=r;provenance[f'{model}_{suite}_{level}']=p
names=['OpenVLA-OFT','π0.5','Cosmos','Fast-WAM'];models=['openvla_oft','pi05','cosmos','fastwam']
grouped('baselines',names,{'Task success':[base[(m,'spatial','I')]['tsr'] for m in models],'Collision avoidance':[base[(m,'spatial','I')]['car'] for m in models]},[BLUE,TEAL])
fol={}
for conf in ['baseline_og10','v19','v20','v21d2','v172']:
 for level in ['L1','L2']:
  p=glob.glob(f'results/safelibero/fol/{conf}/{level}/**/*.json',recursive=True)[0];fol[conf,level]=readresult(p);provenance[f'{conf}_{level}']=p
labels=['Level I\nTask success','Level I\nCollision avoidance','Level II\nTask success','Level II\nCollision avoidance']
grouped('filter_results',labels,{name:[fol[c,l][k] for l in ['L1','L2'] for k in ['tsr','car']] for name,c in [('No filter','baseline_og10'),('Visual filter (v19)','v19')]},[GRAY,TEAL],65)
fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
for ax,l,title in zip(axs,['L1','L2'],['Level I','Level II']):
 for c,label,col,offset in [('baseline_og10','Baseline',GRAY,(5,8)),('v19','RGB ellipsoid',TEAL,(5,8)),('v20','+ corridor',BLUE,(-55,-20)),('v21d2','RGB-D / top 2',ORANGE,(5,8)),('v172','GT oracle','#856098',(5,8))]:
  r=fol[c,l];ax.scatter(r['tsr'],r['car'],s=100,color=col);ax.annotate(label,(r['tsr'],r['car']),xytext=offset,textcoords='offset points',fontsize=10)
 ax.set_xlim(25,48);ax.set_ylim(0,65);ax.set_title(title);ax.set_xlabel('Task success (%)');ax.grid(alpha=.15)
axs[0].set_ylabel('Collision avoidance (%)');savefig('tradeoff')
grouped('memory_results',['Violation','Task success','Safe success'],{'No memory':[75,87.5,15.625],'Memory':[28.125,71.875,50],'Memory deleted':[75,87.5,15.625]},[GRAY,TEAL,ORANGE],105)
grouped('memory_controls',['Hidden hazard','Clean lookalike','Washed','Visible hazard'],{'No memory':[75,0,6.25,28.125],'Memory':[28.125,0,3.125,28.125]},[GRAY,TEAL],100,'Violation rate (%)')
# A replay of recorded kinematic trajectories. No new experiment is run.
records=[json.loads(x) for x in open('experiments/safety_memory_poc/runs/kin_final_v3/episodes.jsonl')]
recs={a:next(r for r in records if r['pair']==0 and r['cell']=='H_POS' and r['arm']==a)['scored'] for a in ['memoryless','memory']}
print('REPLAY', {a:{k:r[k] for k in ['violation','task_success','safe_success']} for a,r in recs.items()})
fig,axs=plt.subplots(1,2,figsize=(10,4.2));lines=[];dots=[]
for ax,(arm,r),col in zip(axs,recs.items(),[ORANGE,TEAL]):
 p=np.asarray(r['traj']);center=np.array(r['rte0']);ax.add_patch(Circle(center[:2],.06,color=ORANGE,alpha=.17));ax.scatter(*center[:2],s=90,marker='s',color=INK);ax.scatter(*r['goal'][:2],s=130,marker='*',color=BLUE)
 ax.set_xlim(-.25,.35);ax.set_ylim(-.22,.22);ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_title('No memory' if arm=='memoryless' else 'Memory');line,=ax.plot([],[],color=col,lw=3);dot,=ax.plot([],[],'o',color=col);lines.append(line);dots.append(dot)
axs[0].set_ylabel('y (m)');fig.suptitle('Recorded kinematic trace · hidden hazard · pair 0',fontsize=15);fig.tight_layout()
with iio.get_writer(A/'memory_replay.mp4',fps=15,macro_block_size=1) as writer:
 for f in range(120):
  for line,dot,r in zip(lines,dots,recs.values()):
   p=np.array(r['traj']);k=max(1,min(len(p),int((f+1)/90*len(p))));line.set_data(p[:k,0],p[:k,1]);dot.set_data([p[k-1,0]],[p[k-1,1]])
  fig.canvas.draw();arr=np.asarray(fig.canvas.buffer_rgba())[:,:,:3];writer.append_data(arr)
  if f==119:Image.fromarray(arr).save(A/'memory_poster.png')
plt.close()
# Editable slide text and shapes.
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
slides=[]
def rgb(h):return RGBColor.from_string(h.lstrip('#'))
def text(s,t,x,y,w,h,size=24,color=INK,bold=False):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
 for i,line in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=rgb(color);p.space_after=Pt(8)
 return sh
def rect(s,x,y,w,h,color=TEAL):
 sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=rgb(color);sh.line.fill.background();return sh
def pic(s,name,x,y,w,h):
 p=A/name;im=Image.open(p);iw,ih=im.size;scale=min(w/iw,h/ih);ww,hh=iw*scale,ih*scale
 return s.shapes.add_picture(str(p),Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))
def slide(title,kicker,notes='',foot=''):
 s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=rgb(BG)
 text(s,kicker.upper(),.5,.24,12,.35,12,TEAL,True);text(s,title,.5,.82,12.3,1.05,30,INK,True)
 text(s,foot,.5,7.04,11.8,.28,10,GRAY);text(s,f'{len(slides)+1:02}',12.35,7.03,.5,.3,11,GRAY)
 s.notes_slide.notes_text_frame.text=notes;slides.append({'title':title,'notes':notes,'footer':foot});return s
def cards(s,items,y=2.35):
 w=(12-.4*(len(items)-1))/len(items)
 for i,(head,body) in enumerate(items):
  x=.65+i*(w+.4);rect(s,x,y,w,2.35,'#E6ECEA');text(s,head,x+.18,y+.22,w-.36,.7,24,TEAL,True);text(s,body,x+.18,y+1.03,w-.36,1.15,20)
# 1
s=slide('Situational safety for robot manipulation','Progress + methodology', 'Class presentation. The two completed studies ask what a robot can avoid from current visual observations and what changes when a safety-relevant fact persists across episodes. Results are simulation-only. Recommended speaking time: 15–20 minutes.','Akanksha Singal · October 2026')
text(s,'Seeing a hazard.\nRemembering what happened.',.65,2.35,6.1,2,34,TEAL,True);pic(s,'baseline_poster.png',7.25,1.9,5.1,4.7)
# 2
s=slide('What counts as a safe rollout?','Experimental setup','SafeLIBERO: spatial, object, goal and long suites, each at hazard levels I and II. The main baseline comparisons here use 50 episodes per task, four tasks, 200 episodes per condition. TSR and CAR are separate marginal rates; neither is the joint safe-success rate. Level I places hazards near the target; Level II places hazards along the movement path.','Source: results/safelibero/policy_baselines/; results/README.md')
cards(s,[('Task success','Did the robot finish?'),('Collision avoidance','Did it avoid contact?'),('Safe success','Did it finish without\na violation?')]);text(s,'4 task suites × 2 hazard levels × 200 episodes',.75,5.45,11.8,.7,28,TEAL)
# 3
s=slide('Task success leaves a safety gap','Policy baselines','Completed spatial Level-I runs, 200 episodes per policy. Cosmos 7.5/10.5, Fast-WAM 45/5, OpenVLA-OFT 42/12, pi0.5 67/14 (TSR/CAR). These describe the evaluated checkpoints and settings, not universal model rankings. Fast-WAM uses the completed fastwam_50ep run. Full baseline board is in the appendix.','SafeLIBERO spatial · Level I · n = 200 per policy')
pic(s,'baselines.png',.7,1.9,11.9,4.8)
# 4
s=slide('Use the images to locate the obstacle','Method 1 · perception','Completed visual filter v19: vision-based obstacle selection and multi-view triangulation produce a 3D obstacle estimate; uncertainty determines ellipsoid geometry, refined using later views. No ground-truth obstacle positions are used in this control path. The diagram summarizes the evaluated obstacle-avoidance mechanism, not the aspirational six-level taxonomy.','Source: fol_safety_filter/filter.py; final campaign report')
pic(s,'baseline_poster.png',.65,2,3.35,3.35)
for x,head,body in [(4.5,'Identify','Which object\nis the obstacle?'),(8.7,'Ground','Where is it\nin 3D?')]:
 rect(s,x,2.45,3.55,2.2,'#E6ECEA');text(s,head,x+.2,2.7,3.1,.5,25,TEAL,True);text(s,body,x+.2,3.38,3.1,1,23)
text(s,'RGB views + task',.8,5.65,4,.5,20);text(s,'3D estimate + uncertainty',7.3,5.65,5,.5,24,TEAL)
# 5
s=slide('Adjust motion near the estimated hazard','Method 1 · action filtering','The implemented geometric filter removes the approaching component near the warning boundary and pushes outward inside the hard boundary. It monitors end-effector and arm checkpoints. v19 uses uncertainty-shaped ellipsoids and multi-frame refinement. This is a diagram of the implemented action correction, not a claim of a verified global safety guarantee or an exact QP solve.','Source: FOLSafetyFilter._apply_cbf and _apply_point_cbf')
cards(s,[('VLA','Propose an action'),('Safety filter','Check geometry.\nCorrect the motion.'),('Robot','Execute the\nfiltered action')]);text(s,'Repeat at each control step',3.9,5.6,7,.6,28,TEAL)
# 6
s=slide('Collision avoidance improves; failures remain','Visual filter results','Matched final campaign, OpenVLA-OFT og10 checkpoint, seed 7. Each condition has 200 episodes. v19 Level I: TSR 40 to 37, CAR 12.5 to 20.5. Level II: TSR stays 37, CAR 6 to 14.5, a gain of 8.5 percentage points. No statistical significance claim. The run working tree was not fully committed; these are recorded experimental results, not a tagged reproducibility claim.','SafeLIBERO spatial · n = 200 per method and level · seed 7')
pic(s,'filter_results.png',.65,1.9,12,4.75)
# 7
s=slide('More avoidance can cost task success','Completed campaign variants','All points are completed 200-episode campaign conditions. v20 adds a target corridor mechanism; v21-D2 uses RGB-D and two obstacle candidates. GT oracle v17.2 uses simulator positions and differs in filter version, so the gap is diagnostic rather than a clean isolated perception ablation. Do not claim the full gap is caused solely by grounding. Unfinished variants are excluded.','Source: results/safelibero/fol/{baseline_og10,v19,v20,v21d2,v172}/')
pic(s,'tradeoff.png',.55,1.9,12.1,4.8)
# 8 videos slot
s=slide('Watch the remaining failure','Recorded simulation · 10-second clips','Same spatial Level-II task 3, episode 0, baseline og10 versus v19. Both recordings show the moka pot obstacle being toppled. These illustrate a failure case, not a successful intervention. The campaign rates on preceding slides summarize all 200 episodes. Playback is the stored 30 fps video, not a timing measurement.','Task: move the bowl from the cabinet to the plate · illustrative episode')
text(s,'No filter',1.2,1.8,5,.5,23,GRAY,True);text(s,'Visual filter (v19)',7.05,1.8,5,.5,23,TEAL,True)
for key,x,url in [('baseline',1.1,'https://drive.google.com/file/d/1-Cw5eGIoiWLHMCl5lAsdyR451hOJYtiz/view?usp=drivesdk'),('filtered',7,'https://drive.google.com/file/d/1Tolf7eE0mZ_5hdTUzoUKQjQ9r6USj0ps/view?usp=drivesdk')]:
 sh=pic(s,key+'_poster.png',x,2.4,4.1,4.1);sh.click_action.hyperlink.address=url
# 9
s=slide('Check what the benchmark actually measures','Evaluation audit','Completed code audit of LIBERO-Safety, a different benchmark from SafeLIBERO. Robot-contact checks compare geometry integer IDs against string names, so that channel cannot fire. Affordance has no constraints. Duplicate predicate keys and success-step accounting also affect scoring. These findings prevent interpreting older 100% CAR values as safety evidence. No claim that the fixes or reruns are complete.','Source: docs/code_review_findings.md · LIBERO-Safety audit')
cards(s,[('Robot contact','ID/name mismatch\nmisses collisions.'),('Affordance','No constraints\nmeans no safety score.'),('Reported results','Do not treat those\n100% rates as evidence.')])
# 10
s=slide('Some hazards outlast the image','Method 2 · persistent facts','Separate proof of concept with a scripted waypoint controller, fixed human-written rules and a JSON fact store. A proximity event marks the tool in a marker-transfer analogue. The next episode resets and randomizes object poses while preserving identity. A hidden mark cannot be recovered from the current observation. No VLM, learned rule induction or physical contamination model is evaluated.','Source: experiments/safety_memory_poc/README.md and RESULTS.md')
cards(s,[('Episode 1','Tool touches\na marked source.'),('Memory','Store MARKED(tool).'),('Episode 2','Keep the marked tool\naway from food.')]);text(s,'Fixed rule: marked tool + ready-to-eat food → keep apart',.75,5.45,12,.8,25,TEAL)
# 11
s=slide('Change the information; keep the controller fixed','Memory experiment design','Four cells: hidden marked source, clean lookalike, marked then washed, visible mark. Four arms share the same scripted policy and barrier projection: memoryless, memory, deleted memory and oracle ledger. The memoryless arm has a perfect current-frame grounder. Observed state features of hidden marked/clean twins are identical in the documented leak check; this is not a claim about all raw camera pixels.','Fixed rules · same policy and filter · four belief sources')
cards(s,[('Hidden','Does memory\nreduce violations?'),('Clean / washed','Does the constraint\nrelease?'),('Visible','Does memory become\nredundant?')]);text(s,'No memory     Memory     Memory deleted     Oracle',.8,5.6,12,.6,24,TEAL)
# 12
s=slide('Memory reduces hidden-hazard violations','Physics proof of concept · Board B','Full scored hidden-positive cell: 32 episodes per arm. Memoryless: violation 24/32, task 28/32, safe 5/32. Memory: violation 9/32, task 23/32, safe 16/32. Deleted memory matches memoryless. Entire board: 640 episodes run, 128 excluded by cell-aware settle criteria, 512 scored. The paired analysis uses a smaller matched subset of 20; do not mix its rates with these full-cell rates. Success is lower with memory; lack of significance does not establish no cost.','Hidden hazard · n = 32 per arm · 512 scored / 640 run across the full board')
pic(s,'memory_results.png',.7,1.9,11.9,4.75)
# 13
s=slide('The effect depends on what can be observed','Memory controls','Chart uses full scored cells, n=32 per arm. On the paired subset, hidden violation is 75% versus 30%; visible violation is 30% in both arms. The interaction is 45 percentage points, 95% CI [25,65], n=20 paired comparisons. Full-cell reset belief remains 6.2%, so do not claim perfect wash release. Memory and oracle match, but both derive their marking signal from the same proximity event.','Full scored cells: n = 32 per arm · paired interaction: 45 pp, 95% CI [25, 65]')
pic(s,'memory_controls.png',.65,1.9,12,4.75)
# 14
s=slide('Replay: the same task with and without memory','Recorded kinematic trace','Animation replays the stored x/y end-effector trajectory for hidden-positive pair 0 in kin_final_v3. This is a projection of a 3D kinematic testbed trace, not MuJoCo video and not the physics board used in the preceding plots. Frame progression interpolates the display time over stored samples, not real-time simulation speed. Square: food location. Shaded disk: 6 cm proximity threshold in the top-down view. Star: goal. Exact 3D violation/success labels come from the recorded episode.','Kinematic testbed · recorded pair 0 · top view · no new rollout')
pic(s,'memory_poster.png',.7,1.95,11.9,4.9)
# 15
s=slide('What we have demonstrated','Progress','Completed work: policy baseline evaluation, visual geometric safety filtering and its campaign, an audit of benchmark scoring, and persistent-fact enforcement in a controlled scripted-policy experiment. These are separate pieces, not an integrated learned memory/VLA system. Visual control still fails in many episodes. Memory improves safe success here but with reduced task success; external perception, learned policy integration and real robots are not established by these results.','Simulation results · visual filter and memory experiments evaluated separately')
cards(s,[('Visual filtering','Higher collision\navoidance in the campaign.'),('Persistent facts','Higher safe success\nin the hidden-hazard test.'),('Remaining limits','Grounding errors.\nControl failures.\nTask-success cost.')])
# 16 full board appendix 2 models easy native table
s=slide('Completed baseline board','Appendix · SafeLIBERO','Each entry is task success / collision avoidance (%), n=200. Values are read from completed JSONs. The pi0.5 Long-I stale markdown entry is replaced with its complete result (58 / 15). Fast-WAM Spatial-I uses the completed 50-episode-per-task run. Cross-policy comparisons are descriptive of these checkpoint configurations. All exact source paths are stored in sources.json.','Each cell: task success / collision avoidance (%) · n = 200')
cols=[.65,3.3,5.65,8,10.25];headers=['Suite / level']+names
for x,t in zip(cols,headers):text(s,t,x,1.95,2.4,.4,17,TEAL,True)
for i,(suite,level) in enumerate((s,l) for s in ['spatial','object','goal','long'] for l in ['I','II']):
 y=2.55+i*.48;text(s,suite.title()+' / '+level,cols[0],y,2.5,.4,17)
 for j,m in enumerate(models):
  r=base.get((m,suite,level));t=f"{r['tsr']:.1f} / {r['car']:.1f}" if r else 'Not available';text(s,t,cols[j+1],y,2.25,.4,17)
prs.save(OUT/'class_progress.pptx');json.dump(slides,open(OUT/'slide_manifest.json','w'),indent=2);json.dump(provenance,open(OUT/'sources.json','w'),indent=2)
(OUT/'speaker_notes.md').write_text('\n\n'.join(f"## {i+1}. {s['title']}\n\n{s['notes']}\n\n{s['footer']}" for i,s in enumerate(slides)))
print('SAVED',len(slides),'slides',OUT/'class_progress.pptx')
