#include "bgfg_vibe.hpp"

bgfg_vibe::bgfg_vibe():R(20),N(20),noMin(2),phi(0)
{
    initDone=false;
    rnd=cv::theRNG();
    ri=0;
}
void bgfg_vibe::init()
{
    for(int i=0;i<rndSize;i++)
    {
        rndp[i]=rnd(phi);
        rndn[i]=rnd(N);
        rnd8[i]=rnd(8);
    }
}
void bgfg_vibe::setphi(int phi)
{
    this->phi=phi;
    for(int i=0;i<rndSize;i++)
    {
        rndp[i]=rnd(phi);
    }
}
void bgfg_vibe::init_model(cv::Mat& firstSample)
{
    std::vector<cv::Mat> channels;
    split(firstSample,channels);
    if(!initDone)
    {
        init();
        initDone=true;
    }
    model=new Model;
    model->fgch= new cv::Mat*[channels.size()];
    model->samples=new cv::Mat**[N];
    model->fg=new cv::Mat(cv::Size(firstSample.cols,firstSample.rows), CV_8UC1);
    for(size_t s=0;s<channels.size();s++)
    {
        model->fgch[s]=new cv::Mat(cv::Size(firstSample.cols,firstSample.rows), CV_8UC1);
        cv::Mat** samples= new cv::Mat*[N];
        for(int i=0;i<N;i++)
        {
            samples[i]= new cv::Mat(cv::Size(firstSample.cols,firstSample.rows), CV_8UC1);
        }
        for(int i=0;i<channels[s].rows;i++)
        {
            int ioff=channels[s].step.p[0]*i;
            for(int j=0;j<channels[0].cols;j++)
            {
                for(int k=0;k<1;k++)
                {
                    (samples[k]->data + ioff)[j]=channels[s].at<uchar>(i,j);
                }
                (model->fgch[s]->data + ioff)[j]=0;

                if(s==0)(model->fg->data + ioff)[j]=0;
            }
        }
        model->samples[s]=samples;
    }
}
void bgfg_vibe::fg1ch(cv::Mat& frame,cv::Mat** samples,cv::Mat* fg)
{
    int step=frame.step.p[0];
    for(int i=1;i<frame.rows-1;i++)
    {
        int ioff= step*i;
        for(int j=1;j<frame.cols-1;j++)
        {
            int count =0,index=0;
            while((count<noMin) && (index<N))
            {
                int dist= (samples[index]->data + ioff)[j]-(frame.data + ioff)[j];
                if(dist<=R && dist>=-R)
                {
                    count++;
                }
                index++;
            }
            if(count>=noMin)
            {
                ((fg->data + ioff))[j]=0;
                int rand= rndp[rdx];
                if(rand==0)
                {
                    rand= rndn[rdx];
                    (samples[rand]->data + ioff)[j]=(frame.data + ioff)[j];
                }
                rand= rndp[rdx];
                int nxoff=ioff;
                if(rand==0)
                {
                    int nx=i,ny=j;
                    int cases= rnd8[rdx];
                    switch(cases)
                    {
                        case 0:
                            //nx--;
                            nxoff=ioff-step;
                            ny--;
                            break;
                        case 1:
                            //nx--;
                            nxoff=ioff-step;
                            ny;
                            break;
                        case 2:
                            //nx--;
                            nxoff=ioff-step;
                            ny++;
                            break;
                        case 3:
                            //nx++;
                            nxoff=ioff+step;
                            ny--;
                            break;
                        case 4:
                            //nx++;
                            nxoff=ioff+step;
                            ny;
                            break;
                        case 5:
                            //nx++;
                            nxoff=ioff+step;
                            ny++;
                            break;
                        case 6:
                            //nx;
                            ny--;
                            break;
                        case 7:
                            //nx;
                            ny++;
                            break;
                    }
                    rand= rndn[rdx];
                    (samples[rand]->data + nxoff)[ny]=(frame.data + ioff)[j];
                }
            }else
            {
                ((fg->data + ioff))[j]=255;
            }
        }
    }
}
cv::Mat* bgfg_vibe::fg(cv::Mat& frame)
{
    std::vector<cv::Mat> channels;
    split(frame,channels);
    for(size_t i=0;i<channels.size();i++)
    {
        fg1ch(channels[i],model->samples[i],model->fgch[i]);
        if(i>0 && i<2)
        {
            bitwise_or(*model->fgch[i-1],*model->fgch[i],*model->fg);
        }
        if(i>=2)
        {
            bitwise_or(*model->fg,*model->fgch[i],*model->fg);
        }
    }
    if(channels.size()==1) return model->fgch[0];
    return model->fg;
}
