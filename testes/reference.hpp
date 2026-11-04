#ifndef TCC_ALTERADA_TEST_REFERENCE_HPP
#define TCC_ALTERADA_TEST_REFERENCE_HPP
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>
#if EXP_PAR
#include <omp.h>
#endif
using namespace std;
using namespace cv;
struct State {
    Mat image;
    array<Mat,5> thr;
    array<Mat,3> grad, out;
    explicit State(const Mat &src):image(src) {
        for(auto &m:thr) m.create(src.size(),CV_8U);
        for(auto &m:grad) m.create(src.size(),CV_32F);
        for(auto &m:out) m.create(src.size(),CV_8U);
    }
    void prepare(const string &algorithm) {
        if(algorithm=="threshold") for(auto &m:thr) image.copyTo(m);
        if(algorithm=="roberts") for(auto &m:grad) m.setTo(0);
    }
};
#include "referencias.inc"
using Function = void (*)(State&);
Function reference(const string &a) {
    if(a=="threshold") return threshold_ref;
    if(a=="equalization") return equalization_ref;
    if(a=="sobel") return sobel_ref;
    if(a=="prewitt") return prewitt_ref;
    if(a=="roberts") return roberts_ref;
    throw runtime_error("algoritmo desconhecido");
}
bool exact(const Mat &a,const Mat &b) {
    if(a.type()!=b.type()||a.size()!=b.size()) return false;
    for(int i=0;i<a.rows;++i)
        if(memcmp(a.ptr(i),b.ptr(i),size_t(a.cols)*a.elemSize())!=0) return false;
    return true;
}
size_t compare(const State &a,const State &b,const string &algorithm) {
    size_t checked=0;
    auto check=[&](const Mat &x,const Mat &y) {
        if(!exact(x,y)) throw runtime_error("DIVERGENCIA: "+algorithm);
        ++checked;
    };
    if(algorithm=="threshold") for(int k=0;k<5;++k) check(a.thr[k],b.thr[k]);
    else if(algorithm=="equalization") check(a.out[0],b.out[0]);
    else for(int k=0;k<3;++k) {check(a.grad[k],b.grad[k]);check(a.out[k],b.out[k]);}
    return checked;
}
void configure(int p,const string &schedule) {
    #if EXP_PAR
    omp_set_dynamic(0); omp_set_num_threads(p);
    omp_set_schedule(schedule=="dynamic"?omp_sched_dynamic:
                     schedule=="guided"?omp_sched_guided:omp_sched_static,0);
    #else
    (void)p;(void)schedule;
    #endif
}
using Case = pair<string, Mat>;
vector<Case> validation_cases() {
    vector<Case> cases;
    for(auto sz:vector<Size>{{1,1},{2,2},{3,3},{5,1},{1,5},{17,13},{65,33}}) {
        for(int level: {0,80,255})
            cases.emplace_back("constant-"+to_string(sz.width)+"x"+to_string(sz.height)+"-"+to_string(level),
                               Mat(sz,CV_8U,Scalar(level)).clone());
    }
    Mat ramp(19,37,CV_8U),step(19,37,CV_8U),impulse(19,37,CV_8U,Scalar(0));
    Mat parent(40,76,CV_8U); RNG rng(0x514343); rng.fill(parent,RNG::UNIFORM,0,256);
    for(int i=0;i<ramp.rows;++i) for(int j=0;j<ramp.cols;++j) {
        ramp.at<uchar>(i,j)=uchar((i*ramp.cols+j)%256);
        step.at<uchar>(i,j)=j<step.cols/2?79:81;
    }
    impulse.at<uchar>(0,0)=255;impulse.at<uchar>(9,18)=255;
    cases.emplace_back("ramp",ramp);cases.emplace_back("threshold-boundary",step);
    cases.emplace_back("impulses",impulse);
    cases.emplace_back("non-contiguous",parent(Rect(1,1,37,19)));
    for(int k=0;k<12;++k) {
        Mat noise(7+k*3,11+k*2,CV_8U); rng.fill(noise,RNG::UNIFORM,0,256);
        cases.emplace_back("random-"+to_string(k),noise);
    }
    for(const string name:{"base","media","alta","uhd4k","lena","building","sudoku"}) {
        string path="tcc-images/"+name+(name=="lena"||name=="building"?".jpg":".png");
        Mat src=imread(path,IMREAD_GRAYSCALE);
        if(src.empty()) throw runtime_error("imagem ausente: "+path);
        cases.emplace_back(name,src);
    }
    return cases;
}
#endif
