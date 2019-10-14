# function_class
a = [1,2]
x, y = a
print(x, y) # 1 2


# function_class
def a(x):
    return '함수로 쓰였습니다' # 같은 파일에 있을시 실행 오류남

class a:
    def __init__(self):
        self.x = 1
        
    def x_plus(self, y):
        return self.x + y
    
    def forward(self, k):
        return self.x_plus(k)
    

plus1 = a()
print(plus1.forward(2))
x = 1
print(a(x))

# two_layer_net
a = ["A", "B"]
a += ['C', 'D']
print(a) # ['A', 'B', 'C', 'D']

