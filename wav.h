#ifndef WAV_H
#define WAV_H
#include <QCoreApplication>
#include <QtEndian>
#include <QDebug>
#include <QFile>
#include <QDataStream>
#include <cmath>
#include "wavchannel.h"

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
    WAV(const QString fileName, qint8 r);


    void readWAV(const QString fileName);
    double first_calculation(double a, QVector<qint16> b);
    void entropia();

    QVector<double> predictStyleCoding(QVector<qint16> canal, QVector<qreal> vectorEPS);
    void differentialStyleCoding();

    void minus_vectors();


    /* ********************* */
    WAVChannel* left;
    WAVChannel* right;

    qreal SystemOfEquations(QVector<qint16>canal);
    qreal divideEPS (QVector<qint16>canal);
    qreal partOne();

    double entropiaL;
    double entropiaR;

    int samples;
    QVector<double> minLsrVector;

    wav_header_t wavHeader;

    qreal EntroBit(QVector<qint16>canal);
    qreal entro_minus(QVector<double> a);
    bool sign(double a);


    bool lusolve(int n, double ** A, double * B, double * X);
    bool ludist(int n, double ** A);

    bool decode(QVector<qint16> channel);

    qint8 r;
    qreal avLsr;

     void pureCoperfieldMagic();
};

#endif // WAV_H
