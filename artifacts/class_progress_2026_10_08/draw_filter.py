from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
BG='#F6F5F1';TEAL='#007F79';ORANGE='#C75B35';INK='#172B3A'
fig,ax=plt.subplots(figsize=(7.5,4.7),facecolor=BG);ax.set_facecolor(BG)
ax.add_patch(Ellipse((.55,.1),.9,1.2,facecolor='#E7C5B8',edgecolor=ORANGE,lw=2))
ax.add_patch(Ellipse((.55,.1),1.6,1.9,facecolor='none',edgecolor=ORANGE,lw=1.5,linestyle='--'))
ax.plot(-.18,.1,'o',color=INK,ms=13)
ax.annotate('',xy=(.42,.75),xytext=(-.18,.1),arrowprops={'arrowstyle':'->','color':ORANGE,'lw':3})
ax.annotate('',xy=(-.18,.75),xytext=(-.18,.1),arrowprops={'arrowstyle':'->','color':TEAL,'lw':3})
ax.text(.22,1.2,'Warning region',color=ORANGE,fontsize=15)
ax.text(.28,.04,'Obstacle',color=INK,fontsize=18)
ax.text(-.32,-.08,'Robot point',color=INK,fontsize=15,ha='right')
ax.text(.23,-.63,'Hard boundary',color=ORANGE,fontsize=15)
ax.text(.9,.73,'Nominal action',color=ORANGE,fontsize=14,ha='left')
ax.text(-.32,.84,'Corrected action',color=TEAL,fontsize=14,ha='right')
ax.set_aspect('equal');ax.set_xlim(-1.5,2.1);ax.set_ylim(-1.05,1.5);ax.axis('off')
plt.savefig(Path('artifacts/class_progress_2026_10_08/assets/filter_geometry.png'),dpi=190,bbox_inches='tight',facecolor=BG)
