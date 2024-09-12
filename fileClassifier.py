import os


class Classifier:
    def __init__(self):
        self.train = {}
        for filename in os.listdir('vocals'):
            if filename.endswith('.wav'):
                self.train[filename] = {}
                self.train[filename]["signal"] = None
                self.train[filename]["result"] = self._name_recognition(self, filename)

    @staticmethod
    def _name_recognition(self, filename):
        vocal = filename.split('_')[0]
        if vocal == 'A' or vocal == 'a':
            return 'a'
        if vocal == 'E' or vocal == 'e':
            return 'e'
        if vocal == 'I' or vocal == 'i':
            return 'i'
        if vocal == 'O' or vocal == 'o':
            return 'o'
        if vocal == 'U' or vocal == 'u':
            return 'u'
        else:
            return 'other'


vocals = Classifier()

for i, j in vocals.train.items():
    print(i + " " + j["result"])
