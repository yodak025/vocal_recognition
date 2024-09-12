
# imports
import librosa
import random
import numpy
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import os
from functools import reduce
import scipy.io.wavfile as wf
import scipy.signal.windows as sig


class NotAsNaiveAsVocal:
    def __init__(self, target_name, protocol):
        self.naive_mel = protocol()
        self.naive_lpc = protocol()

        self.target_name = target_name

        self.train = {}
        for filename in os.listdir('vocals'):
            if filename.endswith('.wav'):
                self.train[filename] = {}
                self.train[filename]["MEL"] = self._extractor_mel("vocals/" + filename)
                self.train[filename]["LPC"] = self._extractor_lpc("vocals/" + filename)
                self.train[filename]["result"] = self._name_recognition(filename)
                if self.train[filename]["result"] == 'other':
                    self.train.pop(filename)
        for filename in os.listdir('Audios'):
            if filename.endswith('.wav'):
                self.train[filename] = {}
                self.train[filename]["MEL"] = self._extractor_mel("Audios/" + filename)
                self.train[filename]["LPC"] = self._extractor_lpc("Audios/" + filename)
                self.train[filename]["result"] = self._name_recognition(filename)
                if self.train[filename]["result"] == 'other':
                    self.train.pop(filename)
        for filename in os.listdir('Mono'):
            if filename.endswith('.wav'):
                self.train[filename] = {}
                self.train[filename]["MEL"] = self._extractor_mel("Mono/" + filename)
                self.train[filename]["LPC"] = self._extractor_lpc("Mono/" + filename)
                self.train[filename]["result"] = self._name_recognition(filename)
                if self.train[filename]["result"] == 'other':
                    self.train.pop(filename)
        items = list(self.train.items())
        random.shuffle(items)

        self.train = dict(items)

    @staticmethod
    def _filter(signal, fs):
        hamming = sig.hamming(0.02 * fs)
        output = []
        for n in range(0, len(signal) - len(hamming), len(hamming)):
            trama = signal[n:n + len(hamming)]

            if numpy.sum((0.5 / len(trama)) * (numpy.abs(numpy.sign(trama[1:]) - numpy.sign(trama[:-1])))) > 0.4:
                output.append(trama)

        return fs, output


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


    def _extractor_mel(self, filename):
        signal, fs = librosa.load(filename)
        # signal, fs = self._filter(signal, fs)

        mfccs = librosa.feature.mfcc(y=signal, n_mfcc=15, sr=fs)
        mel = []
        for i in mfccs:
            mel.append(reduce(lambda x, y: x + y, i)/len(i))        # #media, no se si es lo mejor.
        return mel

    def _extractor_lpc(self, filename):
        # error si estereo
        signal, fs = librosa.load(filename)
        #fs, signal = self._filter(*wf.read(filename))
        n = int(fs * 0.02)
        h = sig.hamming(n)
        k = len(signal) % n

        if k == 0:
            win_num = (len(signal) // n)
        else:
            win_num = (len(signal) // n)+1

        windows = []
        for i in range(win_num-1):
            windows.append(signal[i*n:(i+1)*n] * h)
        if k != 0:
            windows.append(signal[len(signal)-n: len(signal)] * h)
        clpc = list(zip(*list(librosa.lpc(numpy.array(windows), order=12))))

        lpc = []
        for i in clpc:
            lpc.append(reduce(lambda x, y: x + y, i) / len(i))  # #media, no se si es lo mejor.
        return lpc[1:len(clpc)]

    def _fit_mel(self):
        coefficients = []
        results = []
        for i in self.train:
            coefficients.append(self.train[i]["MEL"])
            results.append(self.train[i]["result"])
        self.naive_mel.fit(coefficients, results)

    def _fit_lpc(self):
        coefficients = []
        results = []
        for i in self.train:
            coefficients.append(self.train[i]["LPC"])
            results.append(self.train[i]["result"])
        self.naive_lpc.fit(coefficients, results)

    def _predict_mel(self):
        return self.naive_mel.predict(numpy.array(self._extractor_mel(self.target_name)).reshape(1, -1))

    def _predict_lpc(self):
        return self.naive_lpc.predict(numpy.array(self._extractor_lpc(self.target_name)).reshape(1, -1))

    def _split(self, test_over_1=0.3):
        test_size = int(len(self.train) * test_over_1)
        train_size = len(self.train) - test_size

        print (f"Tamaño de entrenamiento: {train_size}")
        print (f"Tamaño de prueba: {test_size}")
        print(f"Total: {len(self.train)}")

        test = [[], [], []]
        train = [[], [], []]

        for i in range(train_size):
            train[0].append(list(list(self.train.items())[i][1].items())[0][1])
            train[1].append(list(list(self.train.items())[i][1].items())[1][1])
            train[2].append(list(list(self.train.items())[i][1].items())[2][1])

        for i in range(train_size, len(self.train)):
            test[0].append(list(list(self.train.items())[i][1].items())[0][1])
            test[1].append(list(list(self.train.items())[i][1].items())[1][1])
            test[2].append(list(list(self.train.items())[i][1].items())[2][1])
        return train, test

    def fit(self):
        self._fit_mel()
        self._fit_lpc()

    def predict(self):
        print(f"Tu archivo es {self.target_name}")
        print(f"Predicción coeficientes de MEL: {self._predict_mel()}")
        print(f"Predicción coeficientes LPC: {self._predict_lpc()}")

    def accuracy(self, split_rate=0.3):
        mel = GaussianNB()
        lpc = GaussianNB()

        train, test = self._split(split_rate)
        mel.fit(train[0], train[2])
        lpc.fit(train[1], train[2])

        mel_output = list(mel.predict(test[0]))
        lpc_output = list(lpc.predict(test[1]))
        print(f"MEL: {mel_output}")
        print(f"LPC: {lpc_output}")
        print(f"Resultados: {test[2]}")

        print(f"Precisión MEL: {accuracy_score(mel_output, test[2])}")
        print(f"Precisión LPC: {accuracy_score(lpc_output, test[2])}")


if __name__ == "__main__":

    naive = NotAsNaiveAsVocal("U_Mari_nr.wav", GaussianNB)
    naive.fit()
    # print(naive.train)
    naive.predict()
    naive.accuracy(0.25)
    neighborg = NotAsNaiveAsVocal("U_Mari_nr.wav", KNeighborsClassifier)
    neighborg.fit()
    neighborg.predict()
    neighborg.accuracy(0.25)
    neuron = NotAsNaiveAsVocal("U_Mari_nr.wav", MLPClassifier)
    neuron.fit()
    neuron.predict()
    neuron.accuracy(0.25)

