from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

if Path('/root/autodl-tmp/nailfold').exists():
    OUT=Path('/root/autodl-tmp/nailfold/artifacts/report_visuals/technical_architecture.png')
    FONT='/root/autodl-tmp/nailfold/artifacts/report_visuals/fonts/simhei.ttf'
else:
    OUT=Path(r'E:\甲劈微循环\artifacts\report_visuals\technical_architecture.png')
    FONT=r'C:\Windows\Fonts\simhei.ttf'
def F(n): return ImageFont.truetype(FONT,n)
W,H=2560,1580
bg='#F8FAFC'; navy='#18344E'; muted='#65798A'; ink='#253B4F'; blue='#2F6F9F'; green='#2A956B'; amber='#D28A13'; violet='#526CB0'; red='#B94E53'; border='#D6E0E8'; white='#FFFFFF'
im=Image.new('RGB',(W,H),bg); d=ImageDraw.Draw(im)
d.text((150,90),'甲襞微循环智能分析',font=F(56),fill=navy)
d.text((154,166),'从病例图像到可解释的病例级结构化结果',font=F(28),fill=muted)
d.line((150,235,2410,235),fill=border,width=3)

def node(x,y,w,h,num,title,body,accent,fill='#FFFFFF'):
    d.rounded_rectangle((x,y,x+w,y+h),radius=20,fill=fill,outline=border,width=3)
    d.rounded_rectangle((x,y,x+w,y+68),radius=20,fill=accent)
    d.rectangle((x,y+48,x+w,y+68),fill=accent)
    d.text((x+26,y+16),num,font=F(24),fill=white)
    d.text((x+88,y+16),title,font=F(26),fill=white)
    yy=y+102
    for line in body:
        d.text((x+30,yy),line,font=F(22),fill=ink); yy+=42

def arrow(x1,y1,x2,y2):
    d.line((x1,y1,x2,y2),fill='#54758D',width=7)
    d.polygon([(x2,y2),(x2-20,y2-13),(x2-20,y2+13)],fill='#54758D')

node(150,315,480,190,'01','病例输入',['病例文件夹','图片 / 视频 / 报告'],blue,'#EFF6FB')
arrow(630,410,720,410)
node(740,315,480,190,'02','图片帧',['CAPorg*.jpg','逐帧处理 → 病例聚合'],green,'#EFF9F3')
arrow(1220,410,1310,410)
node(1330,315,480,190,'03','实例分割',['YOLOv11l-seg','6类实例 mask'],amber,'#FFF8E9')

d.text((150,610),'分割结果进入三条互补路线',font=F(27),fill=muted)
d.line((1330,505,1330,640),fill='#54758D',width=7)
d.line((430,640,2130,640),fill='#54758D',width=7)
for x in (430,1030,1630):
    d.line((x,640,x,700),fill='#54758D',width=7)
    d.polygon([(x,700),(x-14,680),(x+14,680)],fill='#54758D')

def lane(x,title,subtitle,accent,fill,lines):
    w=500; y=730; h=330
    d.rounded_rectangle((x,y,x+w,y+h),radius=22,fill=fill,outline=accent,width=4)
    d.rectangle((x,y,x+14,y+h),fill=accent)
    d.text((x+36,y+32),title,font=F(29),fill=navy)
    d.text((x+36,y+82),subtitle,font=F(21),fill=accent)
    yy=y+140
    for line in lines:
        d.text((x+42,yy),line,font=F(22),fill=ink); yy+=42
lane(180,'路线 A  几何定量','把结构转成数值',green,'#EAF7F0',['骨架 + 距离变换','端点 / 中心线定位','管径、袢顶直径、袢长','4个连续字段 | 83%–99%'])
lane(780,'路线 B  视觉分类','补充整体与局部语义',violet,'#EEF2FB',['DINOv2视觉特征','分割统计特征 + GBT','颜色、清晰度、病灶等','9个分级字段'])
lane(1380,'路线 C  管袢计数','专门处理实例数量',red,'#FCEFF0',['normal + abnormal实例','多阈值计数','大面积 mask → skeleton分支','输出管袢数'])

d.line((430,1060,430,1150),fill='#54758D',width=7); d.line((1030,1060,1030,1150),fill='#54758D',width=7); d.line((1630,1060,1630,1150),fill='#54758D',width=7)
d.line((430,1150,1630,1150),fill='#54758D',width=7); d.line((1030,1150,1030,1190),fill='#54758D',width=7); d.polygon([(1030,1190),(1016,1170),(1044,1170)],fill='#54758D')
d.rounded_rectangle((560,1195,1460,1325),radius=22,fill=navy)
d.text((610,1220),'病例级汇总',font=F(30),fill=white)
d.text((880,1222),'帧 → 病例  |  17字段  |  JSON',font=F(22),fill='#DCEAF4')

# Expert interpretation layer after model inference.
d.rounded_rectangle((1700,1195,2240,1325),radius=22,fill='#F3ECFB',outline=violet,width=4)
d.text((1740,1220),'甲襞微循环知识库',font=F(26),fill=navy)
d.text((1740,1270),'RAG检索 / 证据片段',font=F(20),fill=violet)
d.line((1970,1325,1970,1375),fill=violet,width=6)
d.polygon([(1970,1375),(1958,1357),(1982,1357)],fill=violet)
d.line((1010,1325,1010,1370),fill='#54758D',width=7)
d.polygon([(1010,1370),(996,1350),(1024,1350)],fill='#54758D')
d.rounded_rectangle((420,1375,2140,1515),radius=22,fill='#FFF5DF',outline=amber,width=4)
d.rectangle((420,1375,434,1515),fill=amber)
d.text((470,1400),'健康专家模型 + RAG知识库',font=F(30),fill=navy)
d.text((470,1455),'检索甲襞微循环知识 → 解读17字段 → 生成个体化健康建议',font=F(22),fill=ink)
d.text((470,1490),'输入仅限已生成的结构化结果；输出为健康教育与就医建议，不替代医生诊断。',font=F(18),fill=muted)
d.text((150,1550),'模型与阈值冻结后，对47例 locked 测试集进行一次性评估；当前结果用于研究验证，不替代临床判断。',font=F(19),fill=muted)
im.save(OUT,dpi=(150,150)); print(OUT)
