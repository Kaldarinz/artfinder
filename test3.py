class A:
    def __init__(self):
        self.a = 1
        self.b = 2

    def __str__(self):
        return f"A(a={self.a}, b={self.b})" 
    
a = A()

def foo():
    global a
    a.b = 3
print(a.b)