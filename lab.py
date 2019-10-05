p='if(a>=b)'
keyword=['if','else','for','while','printf','scanf','elseif','int','double','float',]
operator=['+','-','*','/','<','>','=']
symbol=[',',';','(',')','{','}','"']
constant=['1','2','3','4','5','6','7','8','9','0']
alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
   
my_operator=[]
my_symbol=[]
my_literal=[]
my_keyword=[]
my_identifier=[]
my_constant=[]
size=0
def con(chk):
    if chk in alphabet:
        return True
    else:
        return False
def con1(chk):
    if chk in constant:
        return True
    else:
        return False
while(size!=len(p)):
     if p[size] in operator:
         if p[size+1]=='=':
             temp=p[size]+p[size+1]
             my_operator.append(temp)
             size=size+2
         else:
             my_operator.append(p[size])
             size+size+1
     elif p[size] in symbol:
          if p[size]=='"':
             my_symbol.append('"')
             
             size=size+1
             string=""
             while(p[size]!='"'):
                 if p[size]=='\t':
                
                     p[size].replace("\t", "\\t")
                 elif p[size]=='\n':
                    p[size].replace("\n", "\\n")
                 string=string+p[size]
                 size=size+1
             my_literal.append(string)
             my_symbol.append('"')
             size=size+1
          else:
              my_symbol.append(p[size])
              size=size+1
     elif p[size] in constant:
          string1=""
          while(p[size] in constant):
               string1=string1+p[size]
               size=size+1
          my_constant.append(string1)
     else:
         temp_list=[]
         for k in keyword:
             if p[size]==k[0]:
                 temp_list.append(k)
         temp_list.sort(key = lambda s: len(s),reverse=True)
         c=0
         for j in temp_list:
             l=len(j)
             string2=""
             for m in range(l):
                 string2=string2+p[size+m]
             if(j==string2):
                 my_keyword.append(string2)
                 size=size+l
                 c=1
         if c==0:
             string3=""
           
             while(con(p[size]) or con1(p[size])):
                 string3=string3+p[size]
                 size=size+1
                 
             my_identifier.append(string3)
             


print(my_identifier)
print(my_keyword)
print(my_constant)
print(my_literal)
print(my_symbol)
print("Operator :",end="")
print(my_operator)

         
     
                     
             
