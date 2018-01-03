#include <QCoreApplication>
#include <QtEndian>
#include <QDebug>
#include <QFile>
#include <QDataStream>
#include <cmath>
#include "wav.h"

//Wav Header
//struct wav_header_t
//{
//    char chunkId[4]; //"RIFF" = 0x46464952
//    quint32 chunkSize; //28
//    char format[4]; //"WAVE" = 0x45564157
//    char subchunk1ID[4]; //"fmt " = 0x20746D66
//    quint32 subchunk1Size; //16
//    quint16 audioFormat;
//    quint16 numChannels;
//    quint32 sampleRate;
//    quint32 byteRate;
//    quint16 blockAlign;
//    quint16 bitsPerSample;
//    //[quint16 wExtraFormatBytes;]
//    //[Extra format bytes]
//};

wav_header_t wavHeader;


void write() {

    int tab[] = { 37, 13, 4, -8, -5 };
    int i =0;
    for (i = 0;i<5;i++) {
        qDebug() << "e = " << tab[i];
    }
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    WAV file("ATrain.wav", 3);
        return a.exec();
}
