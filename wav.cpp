#include "wav.h"
#include "wavchannel.h"
#include <QtMath>
#include <algorithm>
#include <iomanip>


WAV::WAV() {}

WAV::WAV(const QString fileName, qint8 r)
{
    right = new WAVChannel;
    left = new WAVChannel;
    this->r = r;
    readWAV(fileName);

   // normal_vectors(); // wektory wypelniane danymi

    double d = ((first_calculation(samples / 2, left->mainSamples) + first_calculation(samples / 2, right->mainSamples)) / 2);
   // minus_vectors();

   differentialStyleCoding();

   pureCoperfieldMagic();
   // decode(right->mainSamples);
   avLsr = (entro_minus(left->predictCoder) + entro_minus(right->predictCoder)) / 2;

}

void WAV::entropia() {
    /* OBLICZANIE PRAWDOPODOBIENSTW */

    for (int j=0;j<65536;j++) {
        left->probabilities[j]=(double)left->occurenceNumber[j]/samples;
        left->probabilitiesMinus[j]=(double)left->occurenceNumberMinus[j]/samples;

        right->probabilities[j]=(double)right->occurenceNumber[j]/samples;
        right->probabilitiesMinus[j]=(double)left->occurenceNumber[j]/samples;
    }

    /* SREDNIA */
    long double srednia = 0;
    long int iloscLiczb = 0;
    for (int j=0;j<65536;j++) {
        srednia = srednia + j*left->occurenceNumber[j] + j*right->occurenceNumber[j];
        iloscLiczb = iloscLiczb +left->occurenceNumber[j] + right->occurenceNumber[j];
        srednia = srednia + j*left->occurenceNumberMinus[j] + j*left->occurenceNumber[j];
        iloscLiczb = iloscLiczb +left->occurenceNumberMinus[j] + left->occurenceNumber[j];
    }
    srednia = (double)srednia/iloscLiczb;

    /* ENTROPIA */
    entropiaL = 0;
    entropiaR = 0;
    for (int j=1;j<65536;j++) {
        if(left->probabilities[j] != 0) // zgodnie z granica dla 0
            entropiaL += left->probabilities[j] * log2((1/left->probabilities[j]));
        if (left->probabilitiesMinus[j] != 0)
            entropiaL += left->probabilitiesMinus[j] * log2((1/left->probabilitiesMinus[j]));
        if (right->probabilities[j] != 0)
            entropiaR += right->probabilities[j] * log2((1/right->probabilities[j]));
        if (right->probabilitiesMinus[j] != 0)
            entropiaR += right->probabilitiesMinus[j] * log2((1/right->probabilitiesMinus[j]));
        if (j == 8009)

            qDebug() << "";
    }
}

void WAV::readWAV(const QString fileName)
{
    if (fileName != "") {
        QFile wavFile(fileName);
        if (!wavFile.open(QIODevice::ReadOnly))
        {
            qDebug() << "Error: Could not open file!";
            return;
        }

        //Read WAV file header
        QDataStream analyzeHeader (&wavFile);
        analyzeHeader.setByteOrder(QDataStream::LittleEndian);
        analyzeHeader.readRawData(wavHeader.chunkId, 4); // "RIFF"
        analyzeHeader >> wavHeader.chunkSize; // File Size
        analyzeHeader.readRawData(wavHeader.format,4); // "WAVE"
        analyzeHeader.readRawData(wavHeader.subchunk1ID,4); // "fmt"
        analyzeHeader >> wavHeader.subchunk1Size; // Format length
        analyzeHeader >> wavHeader.audioFormat; // Format type
        analyzeHeader >> wavHeader.numChannels; // Number of channels
        analyzeHeader >> wavHeader.sampleRate; // Sample rate
        analyzeHeader >> wavHeader.byteRate; // (Sample Rate * BitsPerSample * Channels) / 8
        analyzeHeader >> wavHeader.blockAlign; // (BitsPerSample * Channels) / 8.1
        analyzeHeader >> wavHeader.bitsPerSample; // Bits per sample

        //Print WAV header
//        qDebug() << "File Size: " << wavHeader.chunkSize;
//        qDebug() << "Format Length: " << wavHeader.subchunk1Size;
//        qDebug() << "Format Type: " << wavHeader.audioFormat;
//        qDebug() << "Number of Channels: " << wavHeader.numChannels;
//        qDebug() << "Sample Rate: " <<  wavHeader.sampleRate;
//        qDebug() << "Sample Rate * Bits/Sample * Channels / 8: " << wavHeader.byteRate;
//        qDebug() << "Bits per Sample * Channels / 8.1: " << wavHeader.blockAlign;
//        qDebug() << "Bits per Sample: " << wavHeader.bitsPerSample;

        // szukam kawalkow
        quint32 chunkDataSize = 0;
        QByteArray temp_buff;
        char buff[0x04];
        while (true)
        {
            QByteArray tmp = wavFile.read(0x04);
            temp_buff.append(tmp);
            int idx = temp_buff.indexOf("data");
            if (idx >= 0)
            {
                int lenOfData = temp_buff.length() - (idx + 4);
                memcpy(buff, temp_buff.constData() + idx + 4, lenOfData);
                int bytesToRead = 4 - lenOfData;
                // finish readind size of chunk
                if (bytesToRead > 0)
                {
                    int read = wavFile.read(buff + lenOfData, bytesToRead);
                    if (bytesToRead != read)
                    {
                        qDebug() << "Error: Something awful happens!";
                        return;
                    }
                }
                chunkDataSize = qFromLittleEndian<quint32>((const uchar*)buff); // zmiana kolejności bajtów
                break;
            }
            if (temp_buff.length() >= 8)
            {
                temp_buff.remove(0, 0x04);
            }
        }
        if (!chunkDataSize)
        {
            qDebug() << "Error: Chunk data not found!";
            return;
        }

        //Reading data from the file
        samples = 0;

        int a = 0;
        while (wavFile.read(buff, 0x04) > 0)
        {

            chunkDataSize -= 4;
            ++samples;
            qint16 sampleChannel1 = qFromLittleEndian<qint16>(buff);

            qint16 sampleChannel2 = qFromLittleEndian<qint16>((buff + 2));
            a++;
            if (a<10) {
           // qDebug() << sampleChannel2;

            }
            if (sampleChannel1 > 0) {
                left->occurenceNumber[sampleChannel1]++; //licznik wystapien
            }
            else {
                left->occurenceNumberMinus[sampleChannel1*(-1)]++;
            }
            if (sampleChannel2 > 0) {
                right->occurenceNumber[sampleChannel2]++; //licznik wystapien
            }
            else {
                left->occurenceNumber[sampleChannel2*(-1)]++;
            }

            // podzial kanalow wplywa na umeiszczenie
            switch (wavHeader.numChannels) {
            case 1:
                left->mainSamples.append(sampleChannel1);
                break;
            case 2:
                left->mainSamples.append(sampleChannel1);
                right->mainSamples.append(sampleChannel2);
                break;
            }
            // check the end of the file
            if (chunkDataSize == 0 || chunkDataSize & 0x80000000)
            {
                break;
            }
        }
        qDebug() << "Readed " << samples << " samples...";



        /* SUMA KWADRATOW L I P */

        long double powerSumLeft = 0;
        long double powerSumRight = 0;
        for (int j=1;j<65536;j++) {
            powerSumLeft += pow(static_cast<double>(left->occurenceNumber[j]),2);
            powerSumLeft += pow(static_cast<double>(left->occurenceNumberMinus[j]),2);
            powerSumRight += pow(static_cast<double>(right->occurenceNumber[j]),2);
            powerSumRight += pow(static_cast<double>(left->occurenceNumber[j]),2);
        }

        /* SREDNIA KWADRATOWA Z SUMY KWADRATOW */

        /* TEST WARTOSCI */
//        for (int i=990;i <1000;i++) {
//            qDebug() << "Number: " << i
//                     << " counted: " << occurenceNumberLeft[i]
//                     <<" prob Lewy: "<< probabilitiesleft[i] // test wystapien
//                     <<" prob Prawy: "<< probabilitiesright[i];
//        }


//        qDebug() << "entropia L: " << entropiaL << "entropia R: " << entropiaR;
   //     qDebug() << "Suma kwadratowa L: " << (double)powerSumLeft << "Suma kwadratowa R: " << (double)powerSumRight;
        //write();

        wavFile.close();
    }
}

double WAV::first_calculation(double a, QVector<qint16> b)
{
    double full = 0;
    for (qint32 i = 0; i < a; i++)
    {
        full = (double)(full + (((double)b.at(i) * (double)b.at(i))));
    }

    full = full / a;

    return full;
}


void WAV::differentialStyleCoding() {

  //  qDebug() << left->mainSamples.size();
   if (left->mainSamples.size() == right->mainSamples.size()) {
       for (int i = 0; i < left->mainSamples.size(); i++){
           if (i == 0) {
               left->differentialCoded.append(left->mainSamples.at(i));
               right->differentialCoded.append(right->mainSamples.at(i));
            }
           else {
            left->differentialCoded.append(left->mainSamples.at(i) - left->mainSamples.at(i - 1));
            right->differentialCoded.append(right->mainSamples.at(i) - right->mainSamples.at(i - 1));
            }
       }
   }

   else qDebug() << "Error: number of samples in both channels is not equal.";
}


QVector<double> WAV::predictStyleCoding(QVector<qint16>canal, QVector<double>vectorEPS) {
    qreal sumPredict = 0;
    QVector <double> counters;
    QVector <double> predictValue;
    //vector <int> vectorEPSint(vectorEPS.begin(), vectorEPS.end());

    for (qint32 i = 0; i < samples / 2; i++) {
        sumPredict = 0;
        if (i == 0)
            predictValue.append(canal.at(i));
        else if (i < r)
            predictValue.append(canal.at(i - 1));
        else {
            for (qint8 j = 1; j <= r; j++)
                sumPredict += vectorEPS.at(j - 1) * canal.at(i - j);
            if (sumPredict > 32768 - 1)
                sumPredict = 32768 - 1;
            else if (sumPredict < -32768)
                sumPredict = -32768;
            predictValue.append(qFloor(sumPredict + 0.5));
        }
    }

    for (int i = 0; i < samples / 2; i++) {
        if(i == 0)
            counters.append(canal.at(i));
        else
            counters.append(canal.at(i) - predictValue.at(i));
    }

//    for (int i = 0; i = 10; i ++)
//        qDebug() << counters[i];
    return counters;
}


void WAV::minus_vectors()
{
    for (int i = 0; i < left->mainSamples.size(); i++)
    {
        if (i == 0)
        {
            left->samplesRemove.append(left->mainSamples.at(i));
        }
        else
        {
            left->samplesRemove.append(left->mainSamples.at(i) - left->mainSamples.at(i - 1));
        }
    }
    for (int i = 0; i < right->mainSamples.size(); i++)
    {
        if (i == 0)
        {
            right->samplesRemove.append(right->mainSamples.at(i));
        }
        else
        {
            right->samplesRemove.append(right->mainSamples.at(i) - right->mainSamples.at(i - 1));
        }
    }
}


qreal WAV::SystemOfEquations(QVector<qint16>canal) {
    double **A, *B, *X;
    int n = r;

   // qDebug() << QString::number( fixed, 'f', 10 );
    A = new double *[n];
    B = new double[n];
    X = new double[n];

    for (int i = 0; i < n; i++)
        A[i] = new double[n];

    int N = samples / 2;
    double sumX = 0;
    double sumP = 0;
    QVector<double>matrixX;
    QVector<double>matrixP;
    QVector<double> vectorEPS;

    for (int i = 1; i <= r; i++) {
        for (int j = 1; j <= r; j++) {
            for (int z = r; z < N; z++) {
                sumX += canal.at(z - i) * canal.at(z - j);
                sumP += canal.at(z) * canal.at(z - i);
            }

            if (j == 1)
                matrixP.append(sumP);
            matrixX.append(sumX);
            sumX = 0;
            sumP = 0;
        }
    }

    int counterVector = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = matrixX.at(counterVector);
            counterVector++;
            B[i] = matrixP.at(i);
        }
    }

    if (ludist(n, A) && lusolve(n, A, B, X)) {}
    else qDebug() << "DZIELNIK ZERO\n";

    for (int i = 0; i < r; i++)
        vectorEPS.append(X[i]);

    QVector<double> counters;
    counters = predictStyleCoding(canal, vectorEPS);
    qreal returnEntropia = entro_minus(counters);

    for (int i = 0; i < n; i++)
        delete[] A[i];
    delete[] A;
    delete[] B;
    delete[] X;

    return returnEntropia;
}


qreal WAV::divideEPS (QVector<qint16>canal) {

    QVector<int>si;
    QVector<double>scale;
    QVector<double>descale;
    QVector<double>matrixX;
    QVector<double>matrixP;
    QVector<double>vectorEPS;
    QVector<double>counters;

    int b = 12;
    int k = 120 / r;
    int N = (samples / 2) - (k - 1) * ceil((samples / 2) / k);
    qreal minLsr = 100;
    double Lsr;

    for (int p = 1; p <= k; p++) {
        double **A, *B, *X;
        int n = r;

       // qDebug() << QString::number( fixed, 'f', 10 );
        A = new double *[n];
        B = new double[n];
        X = new double[n];

        for (int i = 0; i < n; i++)
            A[i] = new double[n];

        double sumX = 0;
        double sumP = 0;

        for (int i = 1; i <= r; i++) {
            for (int j = 1; j <= r; j++) {
                for (int z = r + (N * p - N); z < N * p; z++) {
                    sumX += canal.at(z - i) * canal.at(z - j);
                    sumP += canal.at(z) * canal.at(z - i);
                }

                if (j == 1)
                    matrixP.append(sumP);
                matrixX.append(sumX);
                sumX = 0;
                sumP = 0;
            }
        }

        int counterVector = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = matrixX.at(counterVector);
                counterVector++;
                B[i] = matrixP.at(i);
            }
        }

        if (ludist(n, A) && lusolve(n, A, B, X)) {}
        else qDebug() << "DZIELNIK ZERO\n";

        for (int i = 0; i < r; i++)
            vectorEPS.append(X[i]);

        for (int i = 0; i < n; i++)
            delete[] A[i];
        delete[] A;
        delete[] B;
        delete[] X;

        double max = *std::max_element(vectorEPS.constBegin(), vectorEPS.constEnd());
        double min = *std::min_element(vectorEPS.constBegin(), vectorEPS.constEnd());

        if (abs(min) > max)
            max = abs(min);
        max = float(max);

        for (int i = 0; i < r; i++) {
            scale.append(floor(abs(vectorEPS.at(i)) / (max) * (pow(2, b) - 1) + 0.5));
            si.append(sign(vectorEPS.at(i)));
        }

        for (int i = 0; i < r; i++)
            descale.append(((scale.at(i) / (pow(2, b) - 1)) * (max)) * (si.at(i) * 2 - 1));

        counters = predictStyleCoding(canal, descale);
        Lsr = entro_minus(counters) + ((32 + (r - 1) * (b + 1) + 10) / samples);

        if (minLsr > Lsr)
            minLsr = Lsr;

        si.clear();
        scale.clear();
        descale.clear();
        vectorEPS.clear();
        counters.clear();
    }

    return minLsr;
}

qreal WAV::entro_minus(QVector<double> a)
{
    qreal entro = 0;
    QVector <qreal> buffor(262144);
    for (int i = 0; i < a.size(); i++)
    {
        buffor[(static_cast<int>(a.at(i)) + 131072)] += 1;
    }
    for (int i = 0; i < 262144; i++)
    {
        if (buffor.at(i) != 0)
        {
            double p_i = (double)buffor.at(i) / a.size();
            entro = entro + (p_i*log2(p_i));
        }
    }
    return entro *(-1);
}

bool WAV::ludist(int n, double ** A)
{
    const double eps = 1e-12;
    int i, j, k;

    for (k = 0; k < n - 1; k++)
    {
        if (fabs(A[k][k]) < eps) return false;

        for (i = k + 1; i < n; i++)
            A[i][k] /= A[k][k];

        for (i = k + 1; i < n; i++)
            for (j = k + 1; j < n; j++)
                A[i][j] -= A[i][k] * A[k][j];
    }

    return true;
}

bool WAV::lusolve(int n, double ** A, double * B, double * X)
{
    int    i, j;
    double s;
    const double eps = 1e-12;

    X[0] = B[0];

    for (i = 1; i < n; i++)
    {
        s = 0;

        for (j = 0; j < i; j++) s += A[i][j] * X[j];

        X[i] = B[i] - s;
    }

    if (fabs(A[n - 1][n - 1]) < eps) return false;

    X[n - 1] /= A[n - 1][n - 1];

    for (i = n - 2; i >= 0; i--)
    {
        s = 0;

        for (j = i + 1; j < n; j++) s += A[i][j] * X[j];

        if (fabs(A[i][i]) < eps) return false;

        X[i] = (X[i] - s) / A[i][i];
    }

    return true;
}


bool WAV::sign(double a) {
    if (a >= 0)
        return 1;
    else
        return 0;
}



qreal WAV::EntroBit(QVector<qint16>canal) {

    double **A, *B, *X;
    int n = r;
    int N = samples / 2;

   // qDebug() << setprecision(10) << fixed;
    A = new double *[n];
    B = new double[n];
    X = new double[n];

    for (int i = 0; i < n; i++)
        A[i] = new double[n];

    double sumX = 0;
    double sumP = 0;
    QVector<double>matrixX;
    QVector<double>matrixP;
    QVector<double>vectorEPS;

    for (int i = 1; i <= r; i++) {
        for (int j = 1; j <= r; j++) {
            for (int z = r ; z < N ; z++) {
                sumX += canal.at(z - i) * canal.at(z - j);
                sumP += canal.at(z) * canal.at(z - i);
            }

            if (j == 1)
                matrixP.append(sumP);
            matrixX.append(sumX);
            sumX = 0;
            sumP = 0;
        }
    }

    int counterVector = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = matrixX.at(counterVector);
            counterVector++;
            B[i] = matrixP.at(i);
        }
    }

    if (ludist(n, A) && lusolve(n, A, B, X)) {}
    else qDebug() << "DZIELNIK ZERO\n";

    for (int i = 0; i < r; i++)
        vectorEPS.append(X[i]);

    for (int i = 0; i < n; i++)
        delete[] A[i];
    delete[] A;
    delete[] B;
    delete[] X;

   double max = *std::max_element(vectorEPS.constBegin(), vectorEPS.constEnd());
   double min = *std::min_element(vectorEPS.constBegin(), vectorEPS.constEnd());

    if (abs(min) > max)
        max = abs(min);
    max = float(max);

    QVector<int>si;
    QVector<double>scale;
    QVector<double>descale;
    QVector<double> counters;

    double Lsr;
   // double entropia = 0;
    double minLsr = 100;
    int diagramBit = 0;
    for (int b = 5; b <= 16; b++) {

        for (int i = 0; i < r; i++) {
            scale.append(floor(abs(vectorEPS.at(i)) / (max) * (pow(2, b) - 1) + 0.5));
            si.append(sign(vectorEPS.at(i)));
        }

        for (int i = 0; i < r; i++)
            descale.append(((scale.at(i) / (pow(2, b) - 1)) * (max)) * (si.at(i) * 2 - 1));

        counters = predictStyleCoding(canal, descale);
        Lsr = entro_minus(counters) + ((32 + (r - 1) * (b + 1) + 10) / samples);

        if (minLsr > Lsr) {
            minLsr = Lsr;
            diagramBit = b;
        }

        si.clear();
        scale.clear();
        descale.clear();
        counters.clear();
    }
    minLsrVector.append(minLsr);

    return diagramBit;
}


 bool WAV::decode(QVector<qint16> channel) {


    qreal **pA, *pB, *X;
    int N = (samples - 3) / 2;

    int n = r;
    pA = new qreal *[n];
    for (int i = 0; i < n; i++)
        pA[i] = new qreal[n];
    pB = new qreal[n];
    X = new qreal[n];


    qreal x_cnt = 0;
    qreal p_cnt = 0;
    QVector <qreal> x;
    QVector <qreal> p;
    QVector <double> eps;

    for (int i = 1; i <= r; i++)
        for (int j = 1; j <= r; j++) {
            for (int z = r; z < N; z++) {
                x_cnt += channel.at(z - i) * channel.at(z - j);
                p_cnt += channel.at(z) * channel.at(z - i);

            }
            if (j == 1)
                p.append(p_cnt);
            x.append(x_cnt);
            x_cnt = 0;
            p_cnt = 0;
        }

    int cnt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            pA[i][j] = x.at(cnt);
            pB[i] = p.at(i);
            cnt++;
        }
    }

    if (!ludist(n, pA) && !lusolve(n, pA, pB, X))
        qDebug() << "D = 0";

    for (int i = 0; i < r; i++)
        eps.append(X[i]);

    for (int i = 0; i < n; i++)
        delete[] pA[i];
    delete[] pA;
    delete[] pB;
    delete[] X;

    QVector <double> coder = predictStyleCoding(channel,eps);
    QVector <qreal> decoded;
    qreal predict = 0;
    QVector <qreal> value;

    for (qint32 i = 0; i < samples / 2; i++) {
        predict = 0;
        if (i == 0)
            value.append(channel.at(i));
        else if (i <= r)
            value.append(channel.at(i) - channel.at(i - 1));
        else {
            for (qint8 j = 1; j <= r; j++)
                predict += eps.at(j - 1) * channel.at(i - j);
            if (predict > 32768 - 1)
                predict = 32768 - 1;
            else if (predict < -32768)
                predict = -32768;
            value.append(floor(predict + 0.5));
        }
    }

    for (int i = 0; i < samples / 2; i++) {
        if(i == 0)
            decoded.append(value.at(i));
        else if (i < r)
            decoded.append(value.at(i) + channel.at(i - 1));
        else
            decoded.append(coder.at(i) + value.at(i));
    }



    return true;
}


 void WAV::pureCoperfieldMagic() {

     int r1 = r * 0.5; // 0.5 <=> 0.67
     int r2 = r * 0.5; // 0.5 <=> 0.34

     //RIGHT---------------------------------------------------------------------------

     double **A1, *B1, *X1;
     int n1 = r;

  //   qDebug() << setprecision(5) << fixed;
     A1 = new double *[n1];
     B1 = new double[n1];
     X1 = new double[n1];

     for (int i = 0; i < n1; i++)
         A1[i] = new double[n1];

     int N1 = samples / 2;
     double sumX1 = 0;
     double sumP1 = 0;
     QVector<double>matrixXfirst;
     QVector<double>matrixPfirst;
     QVector<double> vectorEPSfirst;

     for (int i = 1; i <= r; i++) {
         for (int j = 1; j <= r; j++) {

             for (int z = r; z < N1; z++) {
                 if(i > r1 && j > r1)
                     sumX1 += left->mainSamples.at(z - i + r1) * left->mainSamples.at(z - j + r1);
                 else if (i > r1)
                     sumX1 += left->mainSamples.at(z - i + r1) * right->mainSamples.at(z - j);
                 else if (j > r1)
                     sumX1 += right->mainSamples.at(z - i) * left->mainSamples.at(z - j + r1);
                 else
                     sumX1 += right->mainSamples.at(z - i) * right->mainSamples.at(z - j);
                 sumP1 += right->mainSamples.at(z) * right->mainSamples.at(z - i);
             }

             if (j == 1)
                 matrixPfirst.append(sumP1);
             matrixXfirst.append(sumX1);
             sumX1 = 0;
             sumP1 = 0;
         }
     }

     int counterVector1 = 0;
     for (int i = 0; i < n1; i++) {
         for (int j = 0; j < n1; j++) {
             A1[i][j] = matrixXfirst.at(counterVector1);
             counterVector1++;
             B1[i] = matrixPfirst.at(i);
         }
     }

     if (ludist(n1, A1) && lusolve(n1, A1, B1, X1)) {}
     else qDebug() << "DZIELNIK ZERO\n";

     for (int i = 0; i < r; i++)
         vectorEPSfirst.append(X1[i]);

     for (int i = 0; i < n1; i++)
         delete[] A1[i];
     delete[] A1;
     delete[] B1;
     delete[] X1;

     //LEFT---------------------------------------------------------------------------

     double **A, *B, *X;
     int n = r;

     //qDebug() << setprecision(5) << fixed;
     A = new double *[n];
     B = new double[n];
     X = new double[n];

     for (int i = 0; i < n; i++)
         A[i] = new double[n];

     int N = samples / 2;
     double sumX = 0;
     double sumP = 0;
     QVector<double>matrixXsecond;
     QVector<double>matrixPseconds;
     QVector<double>vectorEPSsecond;

     for (int i = 1; i <= r; i++) {
         for (int j = 1; j <= r; j++) {
             for (int z = r; z < N; z++) {
                 if (i > r2 && j > r2)
                     sumX1 += left->mainSamples.at(z - i + r2 - 1) * left->mainSamples.at(z - j + r2 - 1);
                 else if (i > r2)
                     sumX += left->mainSamples.at(z - i + r2 - 1) * right->mainSamples.at(z - j);
                 else if (j > r2)
                     sumX += right->mainSamples.at(z - i) * left->mainSamples.at(z - j + r2 - 1);
                 else
                     sumX += right->mainSamples.at(z - i) * right->mainSamples.at(z - j);
                 sumP += right->mainSamples.at(z) * right->mainSamples.at(z - i);
             }

             if (j == 1)
                 matrixPseconds.append(sumP);
             matrixXsecond.append(sumX);
             sumX = 0;
             sumP = 0;
         }
     }

     int counterVector = 0;
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             A[i][j] = matrixXsecond.at(counterVector);
             counterVector++;
             B[i] = matrixPseconds.at(i);
         }
     }

     if (ludist(n, A) && lusolve(n, A, B, X)) {}
     else qDebug() << "DZIELNIK ZERO\n";

     for (int i = 0; i < r; i++)
         vectorEPSsecond.append(X[i]);

     for (int i = 0; i < n; i++)
         delete[] A[i];
     delete[] A;
     delete[] B;
     delete[] X;

     //RIGHT-------------------------------------------------------------------------- -

     double sumPredict1 = 0;
     QVector <double> predictCoderRightfirst;
     QVector <double> predictValueRightfirst;
     for (qint32 i = 0; i < samples / 2; i++) {
         sumPredict1 = 0;
         if (i == 0)
             predictValueRightfirst.append(right->mainSamples.at(i));
         else if (i < r)
             predictValueRightfirst.append(right->mainSamples.at(i) - right->mainSamples.at(i - 1));
         else {
             for (qint32 j = 1; j <= r1; j++)
                 sumPredict1 += vectorEPSfirst.at(j - 1) * right->mainSamples.at(i - j);
             if (sumPredict1 > 32768 - 1)
                 sumPredict1 = 32768 - 1;
             else if (sumPredict1 < -32768)
                 sumPredict1 = -32768;
             predictValueRightfirst.append(floor(sumPredict1 + 0.5));
         }
     }

     for (int i = 0; i < samples / 2; i++) {
         if (i < r)
             predictCoderRightfirst.append(predictValueRightfirst.at(i));
         else
             predictCoderRightfirst.append(right->mainSamples.at(i) - predictValueRightfirst.at(i));

     }

     //RIGHT-------------------------------------------------------------------------- -

     double sumPredict2 = 0;
     QVector <double> predictCoderRightsecond;
     QVector <double> predictValueRightsecond;
     for (qint32 i = 0; i < samples / 2; i++) {
         sumPredict2 = 0;
         if (i == 0)
             predictValueRightsecond.append(right->mainSamples.at(i));
         else if (i < r)
             predictValueRightsecond.append(right->mainSamples.at(i) - right->mainSamples.at(i - 1));
         else {
             for (qint32 j = 0; j <= r2 - 1; j++)
             sumPredict2 += vectorEPSsecond.at(j) * right->mainSamples.at(i - j);
             if (sumPredict2 > 32768 - 1)
                 sumPredict2 = 32768 - 1;
             else if (sumPredict2 < -32768)
                 sumPredict2 = -32768;
             predictValueRightsecond.append(floor(sumPredict2 + 0.5));
         }
     }

     for (int i = 0; i < samples / 2; i++) {
         if (i < r)
             predictCoderRightsecond.append(predictValueRightsecond.at(i));
         else
             predictCoderRightsecond.append(right->mainSamples.at(i) - predictValueRightsecond.at(i));

     }

     //LEFT---------------------------------------------------------------------------

     double sumPredict = 0;
     QVector <double> predictCoderLeftfirst;
     QVector <double> predictValueLeftfirst;
     for (qint32 i = 0; i < samples / 2; i++) {
         sumPredict = 0;
         if (i == 0)
             predictValueLeftfirst.append(left->mainSamples.at(i));
         else if (i < r)
             predictValueLeftfirst.append(left->mainSamples.at(i) - left->mainSamples.at(i - 1));
         else {
             for (qint32 j = 1; j <= r2; j++)
                 sumPredict += vectorEPSsecond.at(j - 1) * left->mainSamples.at(i - j);
             if (sumPredict > 32768 - 1)
                 sumPredict = 32768 - 1;
             else if (sumPredict < -32768)
                 sumPredict = -32768;
             predictValueLeftfirst.append(floor(sumPredict + 0.5));
         }
     }

     for (int i = 0; i < samples / 2; i++) {
         if (i < r)
             predictCoderLeftfirst.append(predictValueLeftfirst.at(i));
         else
             predictCoderLeftfirst.append(left->mainSamples.at(i) - predictValueLeftfirst.at(i));

     }

     //LEFT---------------------------------------------------------------------------

     double sumPredict3 = 0;
     QVector <double> predictCoderLeftsecond;
     QVector <double> predictValueLeftsecond;
     for (qint32 i = 0; i < samples / 2; i++) {
         sumPredict3 = 0;
         if (i == 0)
             predictValueLeftsecond.append(left->mainSamples.at(i));
         else if (i < r)
             predictValueLeftsecond.append(left->mainSamples.at(i) - left->mainSamples.at(i - 1));
         else {
             for (qint32 j = 1; j <= r1; j++)
                 sumPredict3 += vectorEPSfirst.at(j - 1) * left->mainSamples.at(i - j);
             if (sumPredict3 > 32768 - 1)
                 sumPredict3 = 32768 - 1;
             else if (sumPredict3 < -32768)
                 sumPredict3 = -32768;
             predictValueLeftsecond.append(floor(sumPredict3 + 0.5));
         }
     }

     for (int i = 0; i < samples / 2; i++) {
         if (i < r)
             predictCoderLeftsecond.append(predictValueLeftsecond.at(i));
         else
             predictCoderLeftsecond.append(left->mainSamples.at(i) - predictValueLeftsecond.at(i));

     }

     //double sum = 0;
     //for (int i = 0; i < 6; i++) {
     //	qDebug() << vectorEPSfirst.at(i) << endl;
     //	qDebug() << vectorEPSsecond.at(i) << endl;
     //	sum += vectorEPSsecond.at(i);
     //}
     //qDebug() << sum << endl << endl;

     for (int i = 0; i < samples / 2; i++) {
         left->predictCoder.append(predictCoderRightfirst.at(i) + predictCoderLeftfirst.at(i));
         right->predictCoder.append(predictCoderLeftsecond.at(i) + predictCoderRightsecond.at(i));
     }
 }
