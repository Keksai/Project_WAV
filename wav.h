#ifndef WAV_H
#define WAV_H
#include <QCoreApplication>
#include <QtEndian>
#include <QDebug>
#include <QFile>
#include <QDataStream>
#include <cmath>

struct wav_header_t
{
    char chunkId[4]; //"RIFF" = 0x46464952
    quint32 chunkSize; //28
    char format[4]; //"WAVE" = 0x45564157
    char subchunk1ID[4]; //"fmt " = 0x20746D66
    quint32 subchunk1Size; //16
    quint16 audioFormat;
    quint16 numChannels;
    quint32 sampleRate;
    quint32 byteRate;
    quint16 blockAlign;
    quint16 bitsPerSample;
    //[quint16 wExtraFormatBytes;]
    //[Extra format bytes]
};

class WAV
{
public:
    WAV();
    WAV(const QString fileName, const QString fileToSave);

    void readWAV(const QString fileName, const QString fileToSave);
    void entropia();


    quint32 occurenceNumberRight[65536];
    quint32 occurenceNumberMinusRight[65536];

    quint32 occurenceNumberLeft[65536];
    quint32 occurenceNumberMinusLeft[65536];

    double entropiaL;
    double entropiaR;

    int samples;

    double probabilitiesleft[65536];
    double probabilitiesMinusleft[65536];

    double probabilitiesright[65536];
    double probabilitiesMinusright[65536];

    wav_header_t wavHeader;
};

#endif // WAV_H
