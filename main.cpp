#include <QCoreApplication>
#include <QtEndian>
#include <QDebug>
#include <QFile>
#include <QDataStream>
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


int main(int argc, char *argv[])
{
   // QCoreApplication a(argc, argv);

    WAV file("ATrain.wav", 3);
        //return a.exec();
}
