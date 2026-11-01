"""Render the proposed v2 architecture as editable SVG, PNG, PDF and offline HTML.

Run: python docs/design/visualizations/safety-skill-adapter/build.py
Requires CairoSVG for PNG/PDF export. This builds documentation, not robot software.
"""

from pathlib import Path
import html
import json
import re

import cairosvg
from PIL import ImageFont

HERE = Path(__file__).resolve().parent
PALETTE = {
    "memory": ("#eef0fc", "#5354a2"),
    "ground": ("#eaf5f0", "#207460"),
    "frozen": ("#ecf3fc", "#326ba7"),
    "learned": ("#f3edfb", "#7951aa"),
    "check": ("#fff3df", "#a96713"),
    "neutral": ("#f5f5f2", "#68726c"),
}
DETAILS = {}


def esc(value):
    return html.escape(str(value), quote=True)


class Figure:
    def __init__(self, key, title, subtitle, height=1140):
        self.key, self.height = key, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 {height}" '
            f'width="1600" height="{height}" role="img" aria-labelledby="{key}-title {key}-description">',
            f'<title id="{key}-title">{esc(title)}</title>',
            f'<desc id="{key}-description">{esc(subtitle)} Proposed architecture, not experimental results. '
            'Click or focus components in the interactive HTML for details.</desc>',
            '<defs>',
        ]
        for name, (_, color) in PALETTE.items():
            self.parts.append(f'<marker id="{key}-arrow-{name}" viewBox="0 0 10 10" '
                              f'refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                              f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>')
        self.parts += ['</defs>', '<rect width="1600" height="100%" fill="#fffefa"/>']
        self.text(48, 43, "REUSABLE SAFETY SKILLS  /  PROPOSED V2 SYSTEM", 14, "#69736f", weight=600, spacing=2)
        self.text(48, 89, title, 34, "#20312e", weight=650)
        self.text(48, 122, subtitle, 18, "#62716a")

    def text(self, x, y, text, size=19, color="#263d35", weight=400, spacing=0, attrs="", max_width=None):
        if max_width:
            while self.measure(text, size, weight, spacing) > max_width and size > 11:
                size -= .25
        self.parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, sans-serif" '
                          f'font-size="{size}" font-weight="{weight}" fill="{color}" '
                          f'letter-spacing="{spacing}" {attrs}>{esc(text)}</text>')

    @staticmethod
    def measure(text, size, weight=400, spacing=0):
        font=ImageFont.truetype('DejaVuSans-Bold.ttf' if weight>=600 else 'DejaVuSans.ttf',round(size*4))
        return font.getlength(text)/4 + max(0,len(text)-1)*spacing

    def path(self, d, kind="neutral", label=None, at=None, dashed=False, route="", end=True):
        color = PALETTE[kind][1]
        cls = f' class="{route}"' if route else ''
        self.parts.append(f'<g{cls}><path d="{d}" fill="none" stroke="{color}" stroke-width="2.5" '
                          f'stroke-linejoin="round" stroke-linecap="round" '
                          + ('stroke-dasharray="7 6" ' if dashed else '')
                          + (f'marker-end="url(#{self.key}-arrow-{kind})"' if end else '') + '/>')
        if label:
            width=self.measure(label,16)
            self.parts.append(f'<rect x="{at[0]-5}" y="{at[1]-17}" width="{width+10}" height="23" rx="4" fill="#fffefa"/>')
            self.text(*at, label, size=16, color=color)
        self.parts.append('</g>')

    def band(self, x, y, w, h, label):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" '
                          'fill="#fafaf7" stroke="#dce2da" stroke-dasharray="6 5"/>')
        self.text(x+22, y+31, label, 14, "#66746b", weight=600, spacing=1)

    def node(self, key, x, y, w, h, kind, eyebrow, title, lines, detail, route=""):
        fill, color = PALETTE[kind]
        DETAILS[key] = {"title": title, "category": eyebrow, **detail}
        cls = 'component' + (' ' + route if route else '')
        self.parts.append(f'<g class="{cls}" data-node="{key}" role="button" tabindex="0" '
                          f'aria-label="{esc(title)}: show component details">'
                          f'<title>{esc(title)}</title>'
                          f'<rect class="node-surface" x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="13" fill="{fill}" stroke="{color}" stroke-width="1.4"/>')
        self.text(x+20, y+28, eyebrow.upper(), 12, color, weight=650, spacing=1.15,max_width=w-40)
        self.text(x+20, y+60, title, 23, "#24352f", weight=650,max_width=w-40)
        for index, line in enumerate(lines):
            self.text(x+20, y+90+25*index, line, 17, "#52675d",max_width=w-40)
        self.parts.append('</g>')

    def footer(self, text):
        self.parts.append(f'<path d="M48 {self.height-57} H1552" stroke="#dde3db"/>')
        self.text(48, self.height-27, text, 15, "#66736b")
        self.parts.append('</svg>')
        return '\n'.join(self.parts)


def info(purpose, inputs, outputs, rule, section):
    return dict(purpose=purpose, inputs=inputs, outputs=outputs, rule=rule, section=section)


def runtime():
    f = Figure('runtime', 'Skills condition action generation.',
               'The learned adapter changes every flow step. Raw and shielded execution are evaluated separately.', 1170)
    f.band(378, 392, 1145, 300, 'FLOW GENERATION')
    # Edges first, cards second: no line is painted over a component.
    f.path('M344 257 H403', 'memory')
    f.path('M716 257 H775', 'ground')
    # Context enters the adapter, not the base task prompt or integration operator.
    f.path('M945 329 V453', 'ground', 'skill + scene tokens', (971, 386))
    f.path('M344 531 H403', 'frozen', 'original task + images/state', (55, 423))
    f.path('M716 531 H775', 'frozen', 'h_t', (730, 506))
    f.path('M1105 531 H1170', 'learned', 'Δv', (1120, 510))
    f.path('M558 609 V660 H1145 V585 H1170', 'frozen', 'base velocity v_base', (818, 650))
    f.path('M1478 492 H1501 V365 H560 V453', 'frozen', 'next noisy chunk x_t  /  t', (814, 352), dashed=True)
    f.path('M1325 609 V745 H938 V785', 'learned', 'L0 · primary test', (994, 735), route='route-raw')
    f.path('M1325 745 H558 V785', 'check', 'L1 · optional shield', (599, 735), dashed=True, route='route-shield')
    f.path('M1105 851 H1170', 'learned', route='route-raw')
    f.path('M558 921 V955 H1330 V921', 'check', 'checked / repaired command', (790, 947), route='route-shield', dashed=True)
    f.path('M1325 921 V1050 H726', 'neutral', 'actual states + events', (1026, 1038))
    f.path('M195 921 V994', 'neutral')
    f.path('M1478 851 H1550 V160 H362 V469 H344', 'ground', 'fresh observations and binding updates', (640, 152), dashed=True)
    f.path('M362 257 H403', 'ground', end=True)
    f.node('memory',48,185,296,144,'memory','Persistent, editable','Safety-skill memory',
           ['SKILL.md + typed contract','Instructions · limits · applicability'],
           info('Stores reusable requirements, not scene-specific world coordinates.',
                'Human-authored or VLM-drafted instruction, reviewed applicability, supported predicate and numeric limits.',
                'Versioned records that survive a process restart.',
                'Loading or editing a skill does not update adapter weights. Unsupported predicates are rejected.',8))
    f.node('grounding',403,185,313,144,'ground','Refreshed each action','Retrieve + ground',
           ['Bind roles to current objects','Oracle facts / geometry in PoC'],
           info('Determines which skill applies and binds its roles to physical entities.',
                'Skill records, task phase, current facts, geometry and grasp state.',
                'Active requirements, current bindings and evidence/knownness.',
                'Unknown required evidence follows explicit stop/abort handling. Main results use oracle grounding.',9))
    f.node('encoding',775,185,330,144,'frozen','Frozen text encoder','Encode conditioning',
           ['384-D language + 2 parameters','68-value grounded scene vector'],
           info('Converts the instruction and current grounding into adapter inputs.',
                'Normative instruction, two numeric limits, 34 geometry/state values plus 34 knownness bits.',
                'Frozen text embeddings and numeric inputs. The learned token projections belong to the adapter.',
                'Cache instruction embeddings only. Do not cache stale scene geometry. The base task prompt is unchanged.',9))
    f.node('observations',48,453,296,156,'neutral','Current inputs','Task + observation',
           ['Original task stays unchanged','Images · state · grasp / phase','Noisy action x_t and flow time t'],
           info('Provides current task context and robot observations to the frozen policy.',
                'Original task prompt, camera observations, robot state and sampler noise.',
                'The same observation/task interface used by the base checkpoint.',
                'Skill language enters a separate adapter channel. New observations follow each executed prefix.',16))
    f.node('base',403,453,313,156,'frozen','Pretrained weights fixed','Frozen π0.5 backbone',
           ['Predict base flow velocity','Expose action-token features h_t','Recompute on adapted x_t'],
           info('Supplies the pretrained action prior and observation-conditioned features.',
                'Current noisy action chunk x_t, time t, original task and observations.',
                'Base velocity v_base and action-token features h_t.',
                'Base parameters are never optimized. Its predictions still change as the adapter changes x_t.',10))
    f.node('adapter',775,453,330,156,'learned','Learned offline · fixed at runtime','Shared skill adapter',
           ['h_t + x_t + t + skill / scene tokens','2 blocks · width 256 initially','Predict 6-channel pose residual'],
           info('Learns how grounded skill meaning should change continuous action generation.',
                'Frozen features, noisy actions, time, current scene and active language skills.',
                'A residual flow velocity Δv over six pose channels; zero padding for the gripper.',
                'One shared adapter, trained offline. The gripper stays with the base. Outside transport, use the base sampler.',10))
    f.node('integration',1170,453,308,156,'learned','Every flow step','Add + integrate',
           ['v = v_base + Δv_skill','Update x_t toward an action chunk','10 steps · noise → actions'],
           info('Applies the learned residual during generation rather than repairing only the final output.',
                'Base velocity and learned residual, at the same flow time.',
                'An updated noisy chunk; after the final flow step, an adapted action chunk.',
                'Use the implementation convention t=1 noise → t=0 data. Initial Euler dt is −0.1.',13))
    f.node('requirements',48,785,296,136,'neutral','Independent scoring input','Intended requirements',
           ['Externally declared A / B rules','Unaffected by memory deletion'],
           info('Keeps ground-truth obligations independent of the policy’s retrieved memory.',
                'Immutable experiment condition, actual phase and ground-truth facts.',
                'The A/B requirements used to score executed behavior.',
                'Deleting or misretrieving a skill must not delete its evaluator obligation.',7))
    f.node('shield',403,785,313,136,'check','L1 · optional at execution','Checker / repair',
           ['Same fixed backend as F0','Log every replaced command'],
           info('Tests a deployment wrapper separately from the learned model’s raw capability.',
                'The pre-shield adapted chunk, grounded requirements and calibrated predictor.',
                'Accepted, repaired, validated-stop or explicitly aborted action.',
                'Never enable this path in the raw L0 capability comparison. All methods share physical emergency rules.',17), 'route-shield')
    f.node('raw',775,785,330,136,'learned','L0 · primary capability test','Execute raw prefix',
           ['No semantic repair or reranking','Initially execute 1 action'],
           info('Measures what the learned generator can do before a semantic shield rescues it.',
                'The final adapted action chunk.',
                'Its first command, sent directly to simulation.',
                'Common physical emergency termination remains active; this is not a guarantee of safe behavior.',16), 'route-raw')
    f.node('environment',1170,785,308,136,'ground','Existing simulator','LIBERO environment',
           ['Execute command · observe again','Record actual substep events'],
           info('Closes the physical perception/action loop and supplies evaluation observations.',
                'One raw or checked/repaired command.',
                'New images/state, contacts, task-goal and phase evidence.',
                'Simulation pauses during synchronous inference. This measures throughput, not real-time moving-world safety.',16))
    f.node('evaluator',48,994,678,102,'neutral','Evaluation only','Score actual behavior independently',
           ['Task completion + violations + repair use; never used to select raw L0 actions'],
           info('Separates learned compliance from checker-assisted performance.',
                'Actual executed states/events and the externally declared operating requirements.',
                'Safe task success, task success, violations, aborts and repair dependence.',
                'Semantic violations remain latched even if the task subsequently succeeds. No results are claimed here.',20))
    return f.footer('Scope: v2 adapter target. Oracle grounding initially. Only the adapter is trained; all weights are fixed during evaluation.')


def training():
    f=Figure('training','Learn the skill-to-action interface offline.',
             'Checked demonstrations provide targets; gradients update the shared adapter only.',1040)
    f.band(392,397,1170,245,'TRAINING PASS')
    f.path('M350 259 H420','check')
    f.path('M740 259 H810','neutral')
    f.path('M1150 259 H1220','neutral')
    f.path('M980 324 V365 H200 V444','neutral','executed target chunk A*',(270,354))
    f.path('M350 522 H420','frozen','x_t, t',(362,507))
    f.path('M740 522 H810','frozen','h_t',(764,507))
    f.path('M1150 522 H1220','learned','Δv',(1169,507))
    f.path('M580 444 V385 H1380 V444','frozen','v_base  /  no gradient into base',(789,374))
    f.path('M200 611 V676 H1185 V822 H1220','neutral','target velocity = ε − A*',(400,665))
    f.path('M1380 611 V752','learned')
    f.path('M740 823 H775 V701 H875 V611','ground','conditioning',(771,691))
    f.path('M1220 836 H1150','learned','∇φ',(1173,817),dashed=True)
    f.path('M1020 752 V611','learned','update φ only',(1037,708),dashed=True)
    f.node('teacher',48,184,302,140,'check','Teacher collection','F0 + nominal rollouts',
           ['Active rules: checked / repaired','No rule: successful nominal'],
           info('Collects useful supervised actions without training a foundation model.',
                'Existing policy, fixed teacher, training scene and intended requirement.',
                'Actual executed commands and complete trajectories, including failed attempts.',
                'Unsafe nominal proposals are not safe targets merely because their repaired executions succeeded.',12))
    f.node('admission',420,184,320,140,'neutral','Independent acceptance','Validate trajectories',
           ['Original task completes','No scored safety violation'],
           info('Admits positive demonstration targets using the independent scorer.',
                'Executed teacher/nominal trajectory and original goal.',
                'Accepted episodes plus an unfiltered collection-attempt ledger.',
                'Disclose success-conditioned collection; all valid final resets remain in evaluation.',12))
    f.node('dataset',810,184,340,140,'memory','Proposed collection targets','Dataset + windows',
           ['200 train / 50 validation episodes','10-action windows · stride 5'],
           info('Provides balanced supervision from five allowed task/context cells.',
                'Accepted trajectories, skill language, parameters, grounding and provenance.',
                'Masked windows beginning in transport; no windows cross episode boundaries.',
                'Keep all derivatives of a physical reset in the same split. The numbers are proposed targets, not collected data.',11))
    f.node('split',1220,184,330,140,'neutral','Leakage boundary','Reserve transfer tests',
           ['Hold out T2+A and both A+B cells','Hold out parameters + paraphrases'],
           info('Distinguishes fitting demonstrations from supported generalization.',
                'Locked physical-reset, wording and parameter partitions.',
                'Untouched final scenarios for pairings, composition and interpolation.',
                'No final outcomes guide optimizer, architecture, checkpoint or teacher-data selection.',11))
    f.node('noise',48,444,302,167,'neutral','Flow-matching inputs','Sample noise + time',
           ['x_t = t ε + (1 − t) A*','t = 1 is noise; t = 0 is data','Use current observed scene'],
           info('Constructs the noisy input and velocity target for supervised flow matching.',
                'Demonstrated normalized actions A*, Gaussian noise ε and sampled flow time t.',
                'x_t and the velocity target ε − A*.',
                'Follow the inspected OpenPI convention; reversing the sign/time convention breaks learning.',13))
    f.node('trainbase',420,444,320,167,'frozen','No optimizer state / gradients','Frozen base policy',
           ['Observation + original task + x_t','Produces v_base and features h_t','stop_gradient at adapter inputs'],
           info('Computes the base prior while remaining numerically unchanged by optimization.',
                'Training observation, task, noisy demonstrated chunk and flow time.',
                'Detached velocity and feature tensors.',
                'Verify parameter hashes and optimizer membership. Frozen weights still require forward compute.',10))
    f.node('trainadapter',810,444,340,167,'learned','Only trainable network','Shared flow adapter',
           ['Same module as runtime','Zero-initialized output projection','6 pose channels; base gripper kept'],
           info('Learns the shared map from grounded skill context to flow residuals.',
                'Detached base features, x_t, t, language, parameters and grounded scene.',
                'Δv over the six adapted channels.',
                'Train/inference input layout and residual hooks must match exactly. Skill files are not optimized.',10))
    f.node('prediction',1220,444,330,167,'learned','Differentiable residual path','Predicted velocity',
           ['v_prediction = v_base + Δv','Loss uses valid pose channels','Mask terminal padding'],
           info('Combines the frozen prior with the learned residual for supervision.',
                'Base and adapter velocity predictions.',
                'A velocity prediction compared with the demonstration-derived target.',
                'Do not backpropagate through the base or add a safety-loss term that has not been validated.',13))
    f.node('traincontext',420,752,320,144,'ground','Frozen encoder · fresh geometry','Skill + scene context',
           ['Text embeddings cached by revision','Training-only normalizers'],
           info('Supplies the same supported conditioning interface used at runtime.',
                'Training instructions, numeric limits, grounded scene and knownness.',
                'Frozen language vectors and current scene/skill tokens.',
                'Final wording and midpoint parameter values are excluded from training and selection.',9))
    f.node('optimizer',810,752,340,144,'learned','Training only','Update adapter φ',
           ['AdamW · effective batch 32','Save adapter + provenance'],
           info('Updates only the shared adapter from the supervised loss.',
                'Gradients from masked flow matching plus the null-context penalty.',
                'Adapter checkpoint with base/text hashes and selection evidence.',
                'No runtime weight updates. Loss at null context discourages needless changes to nominal behavior.',14))
    f.node('loss',1220,752,330,144,'learned','Supervised objective','Flow-matching loss',
           ['MSE(v_prediction, ε − A*)','+ null-context residual penalty'],
           info('Teaches demonstrated conditional actions and nominal retention.',
                'Velocity prediction, demonstrated velocity target and valid-channel mask.',
                'Scalar loss; gradients flow only to adapter parameters.',
                'A lower training loss is not safety evidence. Evaluate raw closed-loop performance on held-out scenes.',13))
    f.text(49,958,'No gradient updates to skill files, the text encoder, or pretrained π0.5. No online training during final evaluation.',19,'#59675d')
    return f.footer('Dashed purple arrows denote optimizer updates. Other arrows denote data. Targets and budgets are proposed, not measured.')


def grounding():
    f=Figure('grounding','A reusable rule becomes a situated input.',
             'Persist the requirement. Refresh object bindings and geometry. Keep evaluation obligations independent.',1040)
    f.path('M490 290 H575','memory')
    f.path('M1015 290 H1100','ground')
    f.path('M1320 427 V543','ground','bind current entities',(1350,494))
    f.path('M1100 648 H1015','ground')
    f.path('M575 648 H490','learned')
    f.node('ruleexample',48,184,442,243,'memory','Reusable / unchanged template','Persistent requirements',
           ['A: avoid protected overhead regions','B: maintain the payload’s upright axis','Default limits: 0.02 m / 15°','Instruction + applicability + contract','No stored scene-specific coordinates'],
           info('Separates a reusable instruction from episode-specific physical state.',
                'Two reviewed SKILL.md records with supported predicates and instruction text.',
                'Stable templates; supported parameter edits are separate revisions.',
                'The example shows geometric requirements, not a demonstrated physical spill-prevention result.',8))
    f.node('facts',575,184,440,243,'ground','Supplied facts in the main PoC','Observe applicability',
           ['phase = transporting','payload = object_17','target = region_4','overhead_sensitive / orientation_sensitive','These facts are oracle inputs initially'],
           info('Determines whether each stored requirement is relevant now.',
                'Current phase, semantic facts, grasp state and entity identities.',
                'Applicable skill set, not a learned claim about hidden object contents.',
                'Counterfactual contexts toggle facts without changing geometry. Unknown is not silently false.',9))
    f.node('bindingexample',1100,184,450,243,'ground','Episode-local / refreshed','Resolve the bindings',
           ['carried payload → object_17','protected region → region_4','Obtain pose, extent and local up-axis','Measure the grasp transform','Store provenance and knownness'],
           info('Connects symbolic roles to the actual quantities needed for continuous action.',
                'Current objects, geometry and stable grasp evidence.',
                'Fresh relative poses, boundaries, up-axis and validity information.',
                'Rebinding is required in every new scene. A correct sentence with the wrong object binding can still fail.',9))
    f.node('tokenexample',1100,543,450,230,'frozen','Supported conditioning interface','Assemble conditioning',
           ['One 68-value scene vector','One 384-D text embedding per skill','Two normalized parameters per token','No predicate IDs in language model input','Empty set → learned null token'],
           info('Preserves an explicit, testable boundary between semantics and motion learning.',
                'Instruction language plus grounding and normalized limits.',
                'Raw scene and skill inputs; learned projection into tokens happens inside the adapter.',
                'Both parameters are supplied to each token so missing fields do not act as hidden predicate IDs.',9))
    f.node('groundadapter',575,543,440,230,'learned','Shared learned interface','Condition the flow adapter',
           ['Same trained weights for A, B, A+B','New scene → new geometry','Supported edit → new conditioning','No new adapter per file','No retraining on skill load'],
           info('Uses a single learned action interface across the supported memory records.',
                'Current scene/skill tokens plus frozen flow features.',
                'A changed action distribution conditioned on the active requirements.',
                'New supported wording and numeric values are tests of transfer. An entirely new primitive is unsupported.',10))
    f.node('groundtest',48,543,442,230,'neutral','Evidence, not a diagrammed result','Test what actually changes',
           ['Relevant fact → appropriate activation','Irrelevant fact → no extra requirement','Delete record → score obligation remains','Paraphrase → preserve the intended rule','Parameter edit → test changed behavior'],
           info('Tests whether the interface does more than route two arbitrary labels.',
                'Fixed weights and matched physical scenes under controlled context edits.',
                'Observed compliance, task success and conditioning-dependence measurements.',
                'This diagram predicts no rollout outcome. Language must be compared with structured and mismatched controls.',17))
    f.text(48,851,'Illustrative bindings only: object_17 and region_4 are example identifiers, never learned skill IDs.',19,'#5b6e61')
    f.text(48,892,'Grounding extension: cross estimated/oracle geometry with estimated/oracle applicability; keep the adapter fixed.',19,'#5b6e61')
    return f.footer('Memory preserves the requirement; grounding supplies its current meaning in the scene. Oracle inputs remain a stated scope limit.')


def main():
    figures = {'runtime':runtime(), 'training':training(), 'grounding':grounding()}
    for name, svg in figures.items():
        (HERE/f'{name}.svg').write_text(svg)
        cairosvg.svg2png(bytestring=svg.encode(),write_to=str(HERE/f'{name}.png'),scale=1.25)
        cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(HERE/f'{name}.pdf'))
    template=(HERE/'page.template.html').read_text()
    for name, svg in figures.items():
        template=template.replace(f'@@{name.upper()}@@',svg)
    template=template.replace('@@DETAILS@@',json.dumps(DETAILS,ensure_ascii=False).replace('</','<\\/'))
    assert not re.search(r'@@[A-Z]+@@',template)
    (HERE/'architecture.html').write_text(template)
    print(f'Rendered {len(figures)} views, SVG/PNG/PDF exports and architecture.html ({len(DETAILS)} inspectable nodes).')


if __name__=='__main__':
    main()
