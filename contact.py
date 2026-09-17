from PIL import Image,ImageDraw
import pandas as pd
from pathlib import Path
D=pd.read_csv('artifacts/annotations/image_quality_auxiliary.csv');D=D[D.quality_label=='usable'].head(16);ts=[]
for _,r in D.iterrows():
 try:
  im=Image.open(r.image_path).convert('RGB');im.thumbnail((220,160));c=Image.new('RGB',(240,190),'white');c.paste(im,((240-im.width)//2,5));ImageDraw.Draw(c).text((5,170),Path(r.image_path).name,fill='black');ts.append(c)
 except:pass
out=Image.new('RGB',(960,760),'#ddd')
for i,im in enumerate(ts):out.paste(im,((i%4)*240,(i//4)*190))
out.save('artifacts/annotations/usable_contact_sheet_16.jpg');print(len(ts))
