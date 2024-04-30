import ray
import catfish.tpu.utils as u

class ActorObject(object):
    def __init__(self, obj, **kwargs):
        self.obj = obj
        self.meta_data = kwargs
        self._id = u.gen_id()
    
    def __getitem__(self, key):
        return self.meta_data[key]

    def __eq__(self, other):
        if isinstance(other, ActorObject):
            return self._id == other._id
        return NotImplemented

    def __call__(self):
        return obj

    def __del__(self):
        pass

class ObjectHandle(object):
    def __init__(self, **kwargs):
        self._id = u.gen_id()

    def __hash__(self):
        return self._id.__hash__()

    def __eq__(self,other):
        assert isinstance(other, ObjectHandle)
        return self._id

@ray.remote
class TPUActor(object):
    def __init__(self):
        self.heap = {}

    @ray.method(num_returns=1)
    def __contains__(self,x):
        return x in self.heap

    @ray.method(num_returns=1)
    def __getitem__(self, key):
        assert isinstance(key, ObjectHandle)
        return self.heap[key]
    
    @ray.method(num_returns=0)
    def __setitem__(self, key, new_value):
        assert isinstance(key, ObjectHandle)
        self.heap[key] = new_value 

    @ray.method(num_returns=0)
    def __delitem__(self,key):
        assert isinstance(key, ObjectHandle)
        del self.heap[key]

    @ray.method(num_returns=1)
    def __call__(self, f,*args,**kwargs):
        if f not in self:
            raise ValueError("Function is not registered on TPU")
        
        _f = self[f]
        _args = [self.heap[x] for x in args]
        _kwargs = {key:self.heap[x] for key, x in kwargs.items()}

        _output = _f(*_args,**_kwargs)
        
        output = ObjectHandle()
        self[output] = _output
        
        return output

            
            

            
