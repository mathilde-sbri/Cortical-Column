Rin = 430 
RinMax= 251 + 130
RinMin= 251- 130

taum= 15.8 
taumMax = 9.5+ 2.8
taumMin = 9.5- 2.8

gL=1/Rin*1000
gLMax= 1/RinMin*1000
gLMin= 1/RinMax*1000

C = gL*taum
CMax = gLMax*taumMax
CMin = gLMin*taumMin

print(f'gL = {gL} nS, plus = {gLMax} nS, moins = {gLMin} nS')
print(f'C = {C} pF, plus = {CMax} pF, moins = {CMin} pF')