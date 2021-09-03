
class evaluate(object):
    def __init__(self):
        super().__init__()

    def compare_acc(self, predicts, data):
        joint_t, joint_all = 0,0
        for b_idx in range(len(predicts)):
            if self.list_eqa(predicts[b_idx], data[b_idx]['belief_state']):
                joint_t += 1
            joint_all += 1

        return [joint_t, joint_all]

    @staticmethod
    def list_eqa(a, b):
        va =[1 if v in b else 0 for v in a ]
        vb = [ 1 if v in a else 0 for v in b ]
        if (sum(va) == sum(vb)) & (sum(va) == len(a)) & (sum(vb) == len(b)):
            return True
        else:
            return False
