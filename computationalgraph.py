class Gate:
    def __init__(self, name="virtual Gate", is_attached=False):
        self._from = []
        self._to = [] 
        self._value = None
        self._grad = 0
        self._name = name
        self.is_loss = False
        self.is_attached = is_attached
    
    def __repr__(self):
        return '<{0} named {1} at {2}>'.format(
                self.__class__.__name__, self._name, hex(id(self)))
    
        
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
        
    def __add__(self, rhs):
        if isinstance(rhs, Gate):
            return AddGate(self, rhs)
        else:
            return AddGate(self, VariableGate(rhs))
    
    def __mul__(self, rhs):
        if isinstance(rhs, Gate):
            return MulGate(self, rhs)
        else:
            return MulGate(self, VariableGate(rhs))
    

        

class VariableGate(Gate):
    def __init__(self, value, name=None, **kwargs):
        super(VariableGate, self).__init__(**kwargs)
        self._value = value
        self._grad = 0
        if name is None:
            self._name = 'InputGate <{0}>'.format(id(name))
        else:
            self._name = name
    
    def forward(self):
        self._grad = 0
    
    def backward(self):
        return 
class AddGate(Gate):
    def __init__(self, lhs, rhs, name=None, **kwargs):
        assert isinstance(lhs, Gate) and isinstance(rhs, Gate)
        super(AddGate, self).__init__(**kwargs)
        if name is None:
            self._name = 'AddGate for <{0},\n {1}>'.format(lhs._name, rhs._name)
        else:
            self._name = name
        
        self._from.append(lhs)
        self._from.append(rhs)
        
    
    def forward(self):
        self._value = sum([gate._value for gate in self._from])
        self._grad = 0
    
    def backward(self):
        if self._value is None:
            raise Exception('Gate has not been forward yet')

        for gate in self._from:
            gate._grad += self._grad

class MulGate(Gate):
    def __init__(self, lhs, rhs, name=None, **kwargs):
        assert isinstance(lhs, Gate) and isinstance(rhs, Gate)
        super(MulGate, self).__init__(**kwargs)
        if name is None:
            self._name = 'MulGate for <{0},\n {1}>'.format(lhs._name, rhs._name)
        else:
            self._name = name
        self._from.append(lhs)
        self._from.append(rhs)
        self._rhs = rhs
        self._lhs = lhs

        
    def forward(self):
        self._value = self._lhs._value * self._rhs._value
        self._grad = 0
    
    def backward(self):
        if self._value is None:
            raise Exception('Gate has not been forward yet')

        self._lhs._grad += self._grad * self._rhs._value
        self._rhs._grad += self._grad * self._lhs._value
        
        
class MaxGate(Gate):
    def __init__(self, lhs, rhs, name=None, **kwargs):
        assert isinstance(lhs, Gate) and isinstance(rhs, Gate)
        super(MaxGate, self).__init__(**kwargs)
        if name is None:
            self._name = 'MaxGate for <{0},\n {1}>'.format(lhs._name, rhs._name)
        else:
            self._name = name
        self._from.append(lhs)
        self._from.append(rhs)
        self._rhs = rhs
        self._lhs = lhs
        
    def forward(self):
        if self._lhs._value > self._rhs._value:
            self._is_left_greater = True
            self._value = self._lhs._value
        else:
            self._is_left_greater = False
            self._value = self._rhs._value
        self._grad = 0
    
    def backward(self):
        if self._value is None:
            raise Exception('Gate has not been forward yet')

        if self._is_left_greater:
            self._lhs._grad += self._grad
            self._rhs._grad = 0
        else:
            self._rhs._grad += self._grad
            self._lhs._grad = 0
        

        
class SquareGate(Gate):
    def __init__(self, gate, name=None, **kwargs):
        assert isinstance(gate, Gate) 
        super(SquareGate, self).__init__(**kwargs)
        if name is None:
            self._name = 'SquareGate for <{0}>'.format(gate._name)
        else:
            self._name = name
        self._from.append(gate)
        
    def forward(self):
        self._value = self._from[0]._value ** 2
        self._grad = 0
    
    def backward(self):
        if self._value is None:
            raise Exception('Gate has not been forward yet')
        self._from[0]._grad += 2 * self._value * self._grad
 
        
def get_all_gates(loss_gate):
    visited = set()
    visited.add(loss_gate)
    def DFS(gate):
        if not gate._from:
            visited.add(gate)
            return
        for gate_f in gate._from:
            if gate_f in visited:
                continue
            DFS(gate_f)
    DFS(loss_gate)
    return visited

            
def get_topo_order_gates(loss_gate):
    gates = get_all_gates(loss_gate)
    visited = set()
    topo_order = []
    def DFS_topo_order(gate, visited, topo_order):
        if gate in visited:
            return
        visited.add(gate)
#         print(gate)
        for gate_f in gate._from:
            DFS_topo_order(gate_f, visited, topo_order)
        topo_order.append(gate)
    for gate in gates:
        DFS_topo_order(gate, visited, topo_order)
    return topo_order

def forward_propagate(topo_order_gates):
    for gate in topo_order_gates:
#         print(gate._value)
        gate.forward()

def backward_propagate(topo_order_gates):
    topo_order_gates[-1]._grad = 1
    for gate in reversed(topo_order_gates):
        gate.backward()

def gradient_decent(topo_order_gates, lr):
    for gate in topo_order_gates:
        gate._value -= lr * gate._grad if gate.is_attached else 0

x1 = VariableGate(5, name='x1')
x2 = VariableGate(10, name='x2')
x3 = VariableGate(-1, name='x3')
x4 = VariableGate(-1, name='x4')
x5 = VariableGate(2, name='x5')

y = x5 = VariableGate(15, name='y')

w11 = VariableGate(2, name='w11',  is_attached=True)
w12 = VariableGate(1.5, name='w12', is_attached=True)
w13 = VariableGate(-1.5, name='w13', is_attached=True)

b11 = VariableGate(2, name='b11',  is_attached=True)
b12 = VariableGate(1.5, name='b12', is_attached=True)
b13 = VariableGate(-1.5, name='b13', is_attached=True)


f11 = (x1 * w11) + b11
f12 = x2 * w12 + b12
f13 = x3 * w13 + b13

w21 = VariableGate(2, name='w21',  is_attached=True)
w22 = VariableGate(1.5, name='w22', is_attached=True)
w23 = VariableGate(-1.5, name='w33', is_attached=True)

b21 = VariableGate(0.2, name='b21',  is_attached=True)
b22 = VariableGate(0.5, name='b22', is_attached=True)
b23 = VariableGate(-1.5, name='b23', is_attached=True)

f21 = f11 * w21 + b21
f22 = f12 * w22 + b22
f23 = f13 * w23 + b23

f3 = MaxGate(f21, f22)

f3 = MaxGate(f3 ,f23)
diff = f3 + ( y * -1)
loss = SquareGate(diff)
net = get_topo_order_gates(loss)

epoch = 50
for e in range(epoch):
    print('epoch: ', e)
    forward_propagate(net)
    backward_propagate(net)
#     print('w11: ', w11._value)
#     print('w12: ', w12._value)
#     print('w13: ', w13._value)
#     print('b11: ', b11._value)
#     print('b12: ', b12._value)
#     print('w21: ', w21._value)
    print(loss._value)
    gradient_decent(net, lr=0.0001)
print(loss._value)
