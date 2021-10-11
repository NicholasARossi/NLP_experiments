import torch
import unittest

class TestPytorch(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.t1 = torch.rand(2,3)
        cls.t2 = torch.rand(2,3)


    def add_tensors(self):


        # print(t1+t2)

        self.assertEqual(torch.add(self.t1,self.t2),
                         self.t1.add_(self.t2))
        self.assertEqual(0,1)
        # print(t1.add_(t2))
        #
        # z = t1 - t2
        #
        # self.assertEqual(z, torch.sub(t1,t2))



def main():
    x = torch.rand(3,requires_grad = True)
    print(x)
    y =  x + 2
    print(y)
    z = y*y*2
    z = z.mean()
    print(z)

    z.backward()
    print(x.grad)


if __name__ == '__main__':
    main()