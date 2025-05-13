import torch
import abc

class GuidanceSchedule(abc.ABC):
    def __init__(self, scale) -> None:
        super().__init__()
        self.scale = scale
    
    @abc.abstractmethod
    def __call__(self,t):
        pass

class ConstantSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return torch.tensor([self.scale])

class LinearSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return 2 * self.scale * (1-t) 

class InverseLinearSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return 2 * self.scale * t

class CosineSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return self.scale * (torch.cos(torch.pi * t) + 1)

class SineSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return self.scale * ((torch.sin(torch.pi * (t - .5))) + 1)
    
class VShapeSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return 2 * self.scale * torch.where(t < .5, 1-t, t) 

class LambdaShapeSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return 2 * self.scale * torch.where(t > .5, 1-t, t) 

def get_guidance_schedule(name, scale):
    if name == 'constant':
        return ConstantSchedule(scale)
    elif name == 'linear':
        return LinearSchedule(scale)
    elif name == 'inv-linear':
        return InverseLinearSchedule(scale)
    elif name == 'cosine':
        return CosineSchedule(scale)
    elif name == 'sine':
        return SineSchedule(scale)
    elif name == 'V':
        return VShapeSchedule(scale)
    elif name == 'inv-V':
        return LambdaShapeSchedule(scale)
    

def get_guidance_strength(t, config):
    try:
        return get_guidance_schedule(config.guidance.schedule, config.guidance.gamma)(t)[0].item() + 1
    except Exception as e:
        return config.guidance.gamma