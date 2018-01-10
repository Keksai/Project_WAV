#ifndef WAVCHANNEL_H
#define WAVCHANNEL_H

#include <QVector>
#include <QFile>

class WAVChannel
{
public:
    WAVChannel();

    QVector<qint16> mainSamples;

    quint32 occurenceNumber[65536];
    quint32 occurenceNumberMinus[65536];

    double probabilities[65536];
    double probabilitiesMinus[65536];

    QVector<qreal> samplesRemove;
    QVector<qreal> differentialCoded;
    QVector<double>predictCoder;
};

#endif // WAVCHANNEL_H
