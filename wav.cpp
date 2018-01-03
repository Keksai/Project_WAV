#include "wav.h"
#include <QtMath>


WAV::WAV() {}

WAV::WAV(const QString fileName, qint8 r)
{
    this->r = r;
    readWAV(fileName, r);
}

void WAV::entropia() {
    /* OBLICZANIE PRAWDOPODOBIENSTW */

    for (int j=0;j<65536;j++) {
        probabilitiesleft[j]=(double)occurenceNumberLeft[j]/samples;
        probabilitiesMinusleft[j]=(double)occurenceNumberMinusLeft[j]/samples;

        probabilitiesright[j]=(double)occurenceNumberRight[j]/samples;
        probabilitiesMinusright[j]=(double)occurenceNumberMinusRight[j]/samples;
    }

    /* SREDNIA */
    long double srednia = 0;
    long int iloscLiczb = 0;
    for (int j=0;j<65536;j++) {
        srednia = srednia + j*occurenceNumberLeft[j] + j*occurenceNumberRight[j];
        iloscLiczb = iloscLiczb +occurenceNumberLeft[j] + occurenceNumberRight[j];
        srednia = srednia + j*occurenceNumberMinusLeft[j] + j*occurenceNumberMinusRight[j];
        iloscLiczb = iloscLiczb +occurenceNumberMinusLeft[j] + occurenceNumberMinusRight[j];
    }
    srednia = (double)srednia/iloscLiczb;

    /* ENTROPIA */
    entropiaL = 0;
    entropiaR = 0;
    for (int j=1;j<65536;j++) {
        if(probabilitiesleft[j] != 0) // zgodnie z granica dla 0
            entropiaL += probabilitiesleft[j] * log2((1/probabilitiesleft[j]));
        if (probabilitiesMinusleft[j] != 0)
            entropiaL += probabilitiesMinusleft[j] * log2((1/probabilitiesMinusleft[j]));
        if (probabilitiesright[j] != 0)
            entropiaR += probabilitiesright[j] * log2((1/probabilitiesright[j]));
        if (probabilitiesMinusright[j] != 0)
            entropiaR += probabilitiesMinusright[j] * log2((1/probabilitiesMinusright[j]));
        if (j == 8009)

            qDebug() << "";
    }
}

void WAV::readWAV(const QString fileName, qint8 r)
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
        qDebug() << "File Size: " << wavHeader.chunkSize;
        qDebug() << "Format Length: " << wavHeader.subchunk1Size;
        qDebug() << "Format Type: " << wavHeader.audioFormat;
        qDebug() << "Number of Channels: " << wavHeader.numChannels;
        qDebug() << "Sample Rate: " <<  wavHeader.sampleRate;
        qDebug() << "Sample Rate * Bits/Sample * Channels / 8: " << wavHeader.byteRate;
        qDebug() << "Bits per Sample * Channels / 8.1: " << wavHeader.blockAlign;
        qDebug() << "Bits per Sample: " << wavHeader.bitsPerSample;

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
            qDebug() << sampleChannel2;

            }
            if (sampleChannel1 > 0) {
                occurenceNumberLeft[sampleChannel1]++; //licznik wystapien
            }
            else {
                occurenceNumberMinusLeft[sampleChannel1*(-1)]++;
            }
            if (sampleChannel2 > 0) {
                occurenceNumberRight[sampleChannel2]++; //licznik wystapien
            }
            else {
                occurenceNumberMinusRight[sampleChannel2*(-1)]++;
            }

            // podzial kanalow wplywa na umeiszczenie
            switch (wavHeader.numChannels) {
            case 1:
                LeftSamples.append(sampleChannel1);
                break;
            case 2:
                LeftSamples.append(sampleChannel1);
                RightSamples.append(sampleChannel2);
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
            powerSumLeft += pow(static_cast<double>(occurenceNumberLeft[j]),2);
            powerSumLeft += pow(static_cast<double>(occurenceNumberMinusLeft[j]),2);
            powerSumRight += pow(static_cast<double>(occurenceNumberRight[j]),2);
            powerSumRight += pow(static_cast<double>(occurenceNumberMinusRight[j]),2);
        }

        /* SREDNIA KWADRATOWA Z SUMY KWADRATOW */

        /* TEST WARTOSCI */
        for (int i=990;i <1000;i++) {
            qDebug() << "Number: " << i
                     << " counted: " << occurenceNumberLeft[i]
                     <<" prob Lewy: "<< probabilitiesleft[i] // test wystapien
                     <<" prob Prawy: "<< probabilitiesright[i];
        }


        qDebug() << "entropia L: " << entropiaL << "entropia R: " << entropiaR;
        qDebug() << "Suma kwadratowa L: " << (double)powerSumLeft << "Suma kwadratowa R: " << (double)powerSumRight;
        //write();

        wavFile.close();
    }
}


QVector<double> WAV::predictCoder(QVector<int16_t>canal, QVector<double>vectorEPS) {
    qreal sumPredict = 0;
    QVector <double> counters;
    QVector <double> predictValue;
    //vector <int> vectorEPSint(vectorEPS.begin(), vectorEPS.end());

    for (size_t i = 0; i < samples / 2; i++) {
        sumPredict = 0;
        if (i == 0)
            predictValue.append(canal.at(i));
        else if (i < r)
            predictValue.append(canal.at(i - 1));
        else {
            for (size_t j = 1; j <= r; j++)
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

    return counters;
}

