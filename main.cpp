#include <QCoreApplication>
#include <QtEndian>
#include <QDebug>
#include <QFile>
#include <QtCore>
#include <QDataStream>
#include <QFile>
#include <QTextStream>
#include <cmath>
#include "wav.h"

wav_header_t wavHeader;


void write() {

    int tab[] = { 37, 13, 4, -8, -5 };
    int i =0;
    for (i = 0;i<5;i++) {
        qDebug() << "e = " << tab[i];
    }
}


int main()
{
   // QCoreApplication a(argc, argv);




    QStringList titles;
    qreal total[16];

    titles << "ATrain.wav"
           << "BeautySlept.wav"
           << "death2.wav"
           << "experiencia.wav"
           << "chanchan.wav"
           << "female_speech.wav"
           << "FloorEssence.wav"
           << "ItCouldBeSweet.wav"
           << "Layla.wav"
           << "LifeShatters.wav"
           << "macabre.wav"
           << "male_speech.wav"
           << "SinceSlways.wav"
           << "thear1.wav"
           << "TomsDiner.wav"
           << "velvet.wav";


    QFile outFile("log");
    outFile.open(QIODevice::WriteOnly | QIODevice::Append);
    QTextStream ts(&outFile);

    QVector<WAV> tracks (16);

    for (qint8 r = 36; r <= 120; r += 6) {

        qreal wavLsr[16];
        qreal averageTotalLsr = 0;

        for (qint8 i = 0; i < titles.size(); i++ ) {
            WAV track(titles.at(i),r);

            wavLsr[i] = track.avLsr;

            tracks.append(track);
        }
        for (int i = 0; i < 16; i++)
                averageTotalLsr += wavLsr[i];
        qDebug() << "Average Lsr for " << r << "is" <<
                    (averageTotalLsr /16);
        QStringList buf;
        buf << "Average Lsr for " << QString::number(r) << "is" <<
              QString::number((averageTotalLsr /16));
       for (int i = 0; i < buf.size(); ++i)
              ts << buf.at(i) << '\n';


    }


//    WAV track("ATrain.wav", r);
//    WAV track("BeautySlept.wav", r);
//    WAV track("death2.wav", r);
//    WAV track("experiencia.wav", r);
//    WAV track("chanchan.wav", r);




        //return a.exec();
}
