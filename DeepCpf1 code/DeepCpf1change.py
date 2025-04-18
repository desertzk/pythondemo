import numpy as np
from numpy import *
import sys;  

from keras.models import Model
from keras.layers import Input,LSTM
from keras.layers.merge import Multiply
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, AveragePooling1D
import pandas as pd


def main2():
    # print("Usage: python DeepCpf1.py input.txt output.txt")
    # print("input.txt must include 3 columns with single header row")
    # print("\t1st column: sequence index")
    # print("\t2nd column: 34bp target sequence")
    # print("\t3rd column: binary chromain information of the target sequence\n")
    #
    # print("DeepCpf1 currently requires python=2.7.12, theano=0.7.0, keras=0.3.3")
    # print(
    #     "DeepCpf1 available on GitHub requires pre-obtained binary chromatin information (DNase-seq narraow peak data from ENCODE)")
    # print(
    #     "DeepCpf1 web tool, available at http://data.snu.ac.kr/DeepCpf1, provides entire pipeline including binary chromatin accessibility for 125 cell lines\n")
    #
    # if len(sys.argv) < 3:
    #     sys.exit()


    train_data = pd.read_excel('data/41587_2018_BFnbt4061_MOESM39_ESM.xlsx', sheet_name=0)
    test_data = pd.read_excel('data/41587_2018_BFnbt4061_MOESM39_ESM.xlsx', sheet_name=1)
    use_data = train_data[0:14999]
    new_header = test_data.iloc[0]
    test_data = test_data[1:]
    test_data.index = np.arange(0, len(test_data))
    test_data.columns = new_header
    bp34_col = use_data["34 bp synthetic target and target context sequence(4 bp + PAM + 23 bp protospacer + 3 bp)"]
    indel_f = use_data["Indel freqeuncy(Background substracted, %)"]
    SEQ = PREPROCESS_ONE_HOT(bp34_col)

    test_bp34 = test_data["34 bp synthetic target and target context sequence(4 bp + PAM + 23 bp protospacer + 3 bp)"]
    test_indel_f = test_data["Indel freqeuncy(Background substracted, %)"]
    test_SEQ = PREPROCESS_ONE_HOT(test_bp34)

    print("Building models")
    Seq_deepCpf1_Input_SEQ = Input(shape=(34, 4))
    # 这代表80个5*5的卷积核吗
    Seq_deepCpf1_C1 = Convolution1D(80, 5, activation='relu')(Seq_deepCpf1_Input_SEQ)
    Seq_deepCpf1_P1 = AveragePooling1D(2)(Seq_deepCpf1_C1)
    # Flatten 压平 变1维
    Seq_deepCpf1_F = Flatten()(Seq_deepCpf1_P1)
    Seq_deepCpf1_DO1 = Dropout(0.3)(Seq_deepCpf1_F)
    # Dense 全连接层
    Seq_deepCpf1_D1 = Dense(80, activation='relu')(Seq_deepCpf1_DO1)
    Seq_deepCpf1_DO2 = Dropout(0.3)(Seq_deepCpf1_D1)
    Seq_deepCpf1_D2 = Dense(40, activation='relu')(Seq_deepCpf1_DO2)
    Seq_deepCpf1_DO3 = Dropout(0.3)(Seq_deepCpf1_D2)
    Seq_deepCpf1_D3 = Dense(40, activation='relu')(Seq_deepCpf1_DO3)
    Seq_deepCpf1_DO4 = Dropout(0.3)(Seq_deepCpf1_D3)
    Seq_deepCpf1_Output = Dense(1, activation='linear')(Seq_deepCpf1_DO4)
    Seq_deepCpf1 = Model(inputs=[Seq_deepCpf1_Input_SEQ], outputs=[Seq_deepCpf1_Output])

    for layer in Seq_deepCpf1.layers:
        print(layer.output_shape)

    import keras
    # 3 激活模型
    Seq_deepCpf1.compile(optimizer=keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                        loss='mean_squared_error')
    # Seq_deepCpf1.fit(x=SEQ, y=indel_f,epochs=100)


    # Seq_deepCpf1.compile(loss='mse', optimizer=keras.optimizers.sgd(learning_rate=0.005))
    ## 这样2和3 一共三行代码就实现了，所以比tensorflow简单多了
    # Seq_deepCpf1.fit(SEQ, indel_f, epochs=1500, verbose=2, shuffle=False)
    # print("--------------testing-------------------")
    # cost = Seq_deepCpf1.evaluate(test_SEQ, test_indel_f)
    # print("test cost:", cost)

    # 4 训练模型
    print("----------------------training--------------------------")
    for step in range(500):
        cost = Seq_deepCpf1.train_on_batch(SEQ, indel_f)
        print("train cost:", cost)
        # cost = Seq_deepCpf1.fit(SEQ, indel_f, epochs=1, batch_size=1000, verbose=2, shuffle=False)

        if step % 100 == 0:
            print("train cost:", cost)

    # 5 检验模型
    print("--------------testing-------------------")
    cost = Seq_deepCpf1.evaluate(test_SEQ, test_indel_f)
    print("test cost:", cost)


    DeepCpf1_Input_SEQ = Input(shape=(34, 4))
    DeepCpf1_C1 = Convolution1D(80, 5, activation='relu')(DeepCpf1_Input_SEQ)
    DeepCpf1_P1 = AveragePooling1D(2)(DeepCpf1_C1)
    DeepCpf1_F = Flatten()(DeepCpf1_P1)
    DeepCpf1_DO1 = Dropout(0.3)(DeepCpf1_F)
    DeepCpf1_D1 = Dense(80, activation='relu')(DeepCpf1_DO1)
    DeepCpf1_DO2 = Dropout(0.3)(DeepCpf1_D1)
    DeepCpf1_D2 = Dense(40, activation='relu')(DeepCpf1_DO2)
    DeepCpf1_DO3 = Dropout(0.3)(DeepCpf1_D2)
    DeepCpf1_D3_SEQ = Dense(40, activation='relu')(DeepCpf1_DO3)

    DeepCpf1_Input_CA = Input(shape=(1,))
    DeepCpf1_D3_CA = Dense(40, activation='relu')(DeepCpf1_Input_CA)
    DeepCpf1_M = Multiply()([DeepCpf1_D3_SEQ, DeepCpf1_D3_CA])

    DeepCpf1_DO4 = Dropout(0.3)(DeepCpf1_M)
    DeepCpf1_Output = Dense(1, activation='linear')(DeepCpf1_DO4)
    DeepCpf1 = Model(inputs=[DeepCpf1_Input_SEQ, DeepCpf1_Input_CA], outputs=[DeepCpf1_Output])

    # we optimized the mean squared error
    # loss function using the Adam28 optimizer and used dropout29 for the model
    # regularization with a 0.3 dropout rate
    # DeepCpf1.compile(optimizer=keras.optim)



def main():
    # print("Usage: python DeepCpf1.py input.txt output.txt")
    # print("input.txt must include 3 columns with single header row")
    # print("\t1st column: sequence index")
    # print("\t2nd column: 34bp target sequence")
    # print("\t3rd column: binary chromain information of the target sequence\n")
    #
    # print("DeepCpf1 currently requires python=2.7.12, theano=0.7.0, keras=0.3.3")
    # print(
    #     "DeepCpf1 available on GitHub requires pre-obtained binary chromatin information (DNase-seq narraow peak data from ENCODE)")
    # print(
    #     "DeepCpf1 web tool, available at http://data.snu.ac.kr/DeepCpf1, provides entire pipeline including binary chromatin accessibility for 125 cell lines\n")
    #
    # if len(sys.argv) < 3:
    #     sys.exit()


    train_data = pd.read_excel('data/41587_2018_BFnbt4061_MOESM39_ESM.xlsx', sheet_name=0)
    test_data = pd.read_excel('data/41587_2018_BFnbt4061_MOESM39_ESM.xlsx', sheet_name=1)
    use_data = train_data[0:14999]
    new_header = test_data.iloc[0]
    test_data = test_data[1:]
    test_data.index = np.arange(0, len(test_data))
    test_data.columns = new_header
    bp34_col = use_data["34 bp synthetic target and target context sequence(4 bp + PAM + 23 bp protospacer + 3 bp)"]
    indel_f = use_data["Indel freqeuncy(Background substracted, %)"]
    SEQ = PREPROCESS_ONE_HOT(bp34_col)

    test_bp34=test_data["34 bp synthetic target and target context sequence(4 bp + PAM + 23 bp protospacer + 3 bp)"]
    test_indel_f = test_data["Indel freqeuncy(Background substracted, %)"]
    test_SEQ = PREPROCESS_ONE_HOT(test_bp34)

    print("Building models")
    Seq_deepCpf1_Input_SEQ = Input(shape=(34, 4))
    # 这代表80个5*5的卷积核吗
    Seq_deepCpf1_C1 = LSTM(40,input_shape=(34, 4))(Seq_deepCpf1_Input_SEQ)
    # Seq_deepCpf1_C1 = Convolution1D(80, 5, activation='relu')(Seq_deepCpf1_Input_SEQ)

    # Seq_deepCpf1_P1 = AveragePooling1D(2)(Seq_deepCpf1_C1)
    # Flatten 压平 变1维
    # Seq_deepCpf1_F = Flatten()(Seq_deepCpf1_C1)
    # Seq_deepCpf1_DO1 = Dropout(0.3)(Seq_deepCpf1_F)
    # Dense 全连接层
    # Seq_deepCpf1_D1 = LSTM(40)(Seq_deepCpf1_DO1)
    # Seq_deepCpf1_DO2 = Dropout(0.3)(Seq_deepCpf1_D1)
    Seq_deepCpf1_D2 = Dense(40, activation='relu')(Seq_deepCpf1_C1)
    Seq_deepCpf1_DO3 = Dropout(0.3)(Seq_deepCpf1_D2)
    Seq_deepCpf1_D3 = Dense(40, activation='relu')(Seq_deepCpf1_DO3)
    Seq_deepCpf1_DO4 = Dropout(0.3)(Seq_deepCpf1_D3)
    Seq_deepCpf1_Output = Dense(1, activation='linear')(Seq_deepCpf1_DO4)
    Seq_deepCpf1 = Model(inputs=[Seq_deepCpf1_Input_SEQ], outputs=[Seq_deepCpf1_Output])
    import keras
    # 3 激活模型
    Seq_deepCpf1.compile(optimizer=keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                        loss='mean_squared_error')
    # Seq_deepCpf1.fit(x=SEQ, y=indel_f,epochs=100)


    # Seq_deepCpf1.compile(loss='mse', optimizer=keras.optimizers.sgd(learning_rate=0.005))
    ## 这样2和3 一共三行代码就实现了，所以比tensorflow简单多了
    # Seq_deepCpf1.fit(SEQ, indel_f, epochs=1500, verbose=2, shuffle=False)
    # print("--------------testing-------------------")
    # cost = Seq_deepCpf1.evaluate(test_SEQ, test_indel_f)
    # print("test cost:", cost)

    # 4 训练模型
    print("----------------------training--------------------------")
    for step in range(200):
        cost = Seq_deepCpf1.train_on_batch(SEQ, indel_f)
        # print("train cost:", cost)
        # cost = Seq_deepCpf1.fit(SEQ, indel_f, epochs=1, batch_size=1000, verbose=2, shuffle=False)

        if step % 100 == 0:
            print("train cost:", cost)

    # 5 检验模型
    print("--------------testing-------------------")
    cost = Seq_deepCpf1.evaluate(test_SEQ, test_indel_f)
    print("test cost:", cost)


    DeepCpf1_Input_SEQ = Input(shape=(34, 4))
    DeepCpf1_C1 = Convolution1D(80, 5, activation='relu')(DeepCpf1_Input_SEQ)
    DeepCpf1_P1 = AveragePooling1D(2)(DeepCpf1_C1)
    DeepCpf1_F = Flatten()(DeepCpf1_P1)
    DeepCpf1_DO1 = Dropout(0.3)(DeepCpf1_F)
    DeepCpf1_D1 = Dense(80, activation='relu')(DeepCpf1_DO1)
    DeepCpf1_DO2 = Dropout(0.3)(DeepCpf1_D1)
    DeepCpf1_D2 = Dense(40, activation='relu')(DeepCpf1_DO2)
    DeepCpf1_DO3 = Dropout(0.3)(DeepCpf1_D2)
    DeepCpf1_D3_SEQ = Dense(40, activation='relu')(DeepCpf1_DO3)

    DeepCpf1_Input_CA = Input(shape=(1,))
    DeepCpf1_D3_CA = Dense(40, activation='relu')(DeepCpf1_Input_CA)
    DeepCpf1_M = Multiply()([DeepCpf1_D3_SEQ, DeepCpf1_D3_CA])

    DeepCpf1_DO4 = Dropout(0.3)(DeepCpf1_M)
    DeepCpf1_Output = Dense(1, activation='linear')(DeepCpf1_DO4)
    DeepCpf1 = Model(inputs=[DeepCpf1_Input_SEQ, DeepCpf1_Input_CA], outputs=[DeepCpf1_Output])

    # we optimized the mean squared error
    # loss function using the Adam28 optimizer and used dropout29 for the model
    # regularization with a 0.3 dropout rate
    # DeepCpf1.compile(optimizer=keras.optim)




def main1():
    print("Usage: python DeepCpf1.py input.txt output.txt")
    print ("input.txt must include 3 columns with single header row")
    print ("\t1st column: sequence index")
    print ("\t2nd column: 34bp target sequence")
    print ("\t3rd column: binary chromain information of the target sequence\n"   )

    print ("DeepCpf1 currently requires python=2.7.12, theano=0.7.0, keras=0.3.3")
    print ("DeepCpf1 available on GitHub requires pre-obtained binary chromatin information (DNase-seq narraow peak data from ENCODE)")
    print ("DeepCpf1 web tool, available at http://data.snu.ac.kr/DeepCpf1, provides entire pipeline including binary chromatin accessibility for 125 cell lines\n" )
    
    if len(sys.argv) < 3:
        sys.exit()
    
    print ("Building models")
    Seq_deepCpf1_Input_SEQ = Input(shape=(34,4))
    # 这代表80个5*5的卷积核吗
    Seq_deepCpf1_C1 = Convolution1D(80, 5, activation='relu')(Seq_deepCpf1_Input_SEQ)
    Seq_deepCpf1_P1 = AveragePooling1D(2)(Seq_deepCpf1_C1)
    # Flatten 压平 变1维
    Seq_deepCpf1_F = Flatten()(Seq_deepCpf1_P1)
    Seq_deepCpf1_DO1= Dropout(0.3)(Seq_deepCpf1_F)
    # Dense 全连接层
    Seq_deepCpf1_D1 = Dense(80, activation='relu')(Seq_deepCpf1_DO1)
    Seq_deepCpf1_DO2= Dropout(0.3)(Seq_deepCpf1_D1)
    Seq_deepCpf1_D2 = Dense(40, activation='relu')(Seq_deepCpf1_DO2)
    Seq_deepCpf1_DO3= Dropout(0.3)(Seq_deepCpf1_D2)
    Seq_deepCpf1_D3 = Dense(40, activation='relu')(Seq_deepCpf1_DO3)
    Seq_deepCpf1_DO4= Dropout(0.3)(Seq_deepCpf1_D3)
    Seq_deepCpf1_Output = Dense(1, activation='linear')(Seq_deepCpf1_DO4)
    Seq_deepCpf1 = Model(inputs=[Seq_deepCpf1_Input_SEQ], outputs=[Seq_deepCpf1_Output])
    
    DeepCpf1_Input_SEQ = Input(shape=(34,4))
    DeepCpf1_C1 = Convolution1D(80, 5, activation='relu')(DeepCpf1_Input_SEQ)
    DeepCpf1_P1 = AveragePooling1D(2)(DeepCpf1_C1)
    DeepCpf1_F = Flatten()(DeepCpf1_P1)
    DeepCpf1_DO1= Dropout(0.3)(DeepCpf1_F)
    DeepCpf1_D1 = Dense(80, activation='relu')(DeepCpf1_DO1)
    DeepCpf1_DO2= Dropout(0.3)(DeepCpf1_D1)
    DeepCpf1_D2 = Dense(40, activation='relu')(DeepCpf1_DO2)
    DeepCpf1_DO3= Dropout(0.3)(DeepCpf1_D2)
    DeepCpf1_D3_SEQ = Dense(40, activation='relu')(DeepCpf1_DO3)
    
    DeepCpf1_Input_CA = Input(shape=(1,))
    DeepCpf1_D3_CA = Dense(40, activation='relu')(DeepCpf1_Input_CA)
    DeepCpf1_M = Multiply()([DeepCpf1_D3_SEQ, DeepCpf1_D3_CA])
    
    DeepCpf1_DO4= Dropout(0.3)(DeepCpf1_M)
    DeepCpf1_Output = Dense(1, activation='linear')(DeepCpf1_DO4)
    DeepCpf1 = Model(inputs=[DeepCpf1_Input_SEQ, DeepCpf1_Input_CA], outputs=[DeepCpf1_Output])
    
    print ("Loading weights for the models")
    Seq_deepCpf1.load_weights('weights/Seq_deepCpf1_weights.h5')
    DeepCpf1.load_weights('weights/DeepCpf1_weights.h5')
    
    print ("Loading test data")
    FILE = open(sys.argv[1], "r")
    data = FILE.readlines()
    SEQ, CA = PREPROCESS(data)
    FILE.close()
    
    print ("Predicting on test data")
    Seq_deepCpf1_SCORE = Seq_deepCpf1.predict([SEQ], batch_size=50, verbose=0)
    DeepCpf1_SCORE = DeepCpf1.predict([SEQ, CA], batch_size=50, verbose=0) * 3
    
    print ("Saving to " + sys.argv[2])
    OUTPUT = open(sys.argv[2], "w")
    for l in range(len(data)):
        if l == 0:
            OUTPUT.write(data[l].strip())
            OUTPUT.write("\tSeq-deepCpf1 Score\tDeepCpf1 Score\n")
        else:
            OUTPUT.write(data[l].strip())
            OUTPUT.write("\t%f\t%f\n" % (Seq_deepCpf1_SCORE[l-1], DeepCpf1_SCORE[l-1]))
    OUTPUT.close()
    
def PREPROCESS(lines):
    data_n = len(lines) - 1
    SEQ = zeros((data_n, 34, 4), dtype=int)
    CA = zeros((data_n, 1), dtype=int)
    
    for l in range(1, data_n+1):
        data = lines[l].split()
        seq = data[1]
        for i in range(34):
            if seq[i] in "Aa":
                SEQ[l-1, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l-1, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l-1, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l-1, i, 3] = 1
        CA[l-1,0] = int(data[2])

    return SEQ, CA


def PREPROCESS_ONE_HOT(train_data):
    data_n = len(train_data)
    SEQ = zeros((data_n, 34, 4), dtype=int)
    # CA = zeros((data_n, 1), dtype=int)

    for l in range(0, data_n):

        seq = train_data[l]
        for i in range(34):
            if seq[i] in "Aa":
                SEQ[l, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l , i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l, i, 3] = 1
        # CA[l - 1, 0] = int(data[2])

    return SEQ



if __name__ == '__main__':
        main2()
        
