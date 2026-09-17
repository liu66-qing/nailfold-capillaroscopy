from pathlib import Path
p=Path('/root/nailfold/scripts/field_labels.py')
s=p.read_text(); old="    'microthrombus': {'未见': '无'},"; new="    'microthrombus': {'未见': '无', '不见': '无'},\n    'hemorrhage': {'未见': '无', '不见': '无'},\n    'exudation': {'未见': '无', '不见': '无'},\n    'blood_color': {'淡红色': '淡红', '浅红色': '浅红', '暗红色': '暗红'},"; s=s.replace(old,new); p.write_text(s)
