//
// Created by mzc on 2020/9/10.
//
#include "../include/Objection.h"
#include "../include/include.h"
#include "Algorithm_Objection_3D.h"
#include "time.h"
using namespace std;
Objection::Objection(cv::Rect Box, int ID){
    Aera_Objection_R=Area_limit(Box);//初始化RGB图像目标框
    ClassID=ID;
    Classname=classNamesVec[ID];//目标类别名称
    Position_Transform Start_point(array<int,2>{Aera_Objection_R.x,Aera_Objection_R.y}, true);//转换彩色图的坐标到深度图
    array<int,2> Start_pix=Start_point.Report_Depth_Pix();//转换
    Box.x=Start_pix.at(0);Box.y=Start_pix.at(1);//更新矩形框
    Area_Objection_D=Area_limit(Box);//越界限制下初始化深度图下的矩形区域
    DealRect();//处理矩形框 稀疏化 获取中心点 聚类
    CheckStartPoint();
    Transform_ImgtoCam();//将中心点坐标转换
    ostringstream ss;
    if (Enable==true)
        ss << "("<<static_cast<int>(Point_Camera.at(0))<<","<<static_cast<int>(Point_Camera.at(1))<<","<<static_cast<int>(Point_Camera.at(2)) <<")";
    else
        ss<<"Null";
    putText(color_mat, ss.str(), cv::Point(Point_img.at(0), Point_img.at(1)), 0, 0.3, cv::Scalar(0, 255, 0));
}
cv::Rect  Objection::Area_limit(cv::Rect Box) {
    cv::Rect Obj;
    Obj.x=(Box.x<0 ? 0:Box.x);//起始点越界检测
    Obj.y=(Box.y<0 ? 0:Box.y);
    Obj.height=(Box.y<0 ? Box.height+Box.y:Box.height); //起始点越界修正目标的宽度和高度
    Obj.width=(Box.x<0 ? Box.width+Box.x:Box.width);
    Obj.height=(Obj.height+Obj.y>(HeightCam-1) ? (HeightCam-1)-Obj.y:Obj.height);//目标框大小越界检测
    Obj.width=(Obj.width+Obj.x>(WidthCam-1) ? (WidthCam-1)-Obj.x:Obj.width);
    return Obj;
}
void Objection::CheckStartPoint() {
    array<float,2> Pre_point;
   Pre_point.at(0)=Aera_Objection_R.x+Aera_Objection_R.width/2 ;
   Pre_point.at(1)=Aera_Objection_R.y+Aera_Objection_R.height/2;
//   cout<<Classname<<": Point_img:"<<Point_img.at(0)<<"  "<<Point_img.at(1)<<endl;
//   if (Enable== false) {
   Point_img.at(0)=Pre_point.at(0);
   Point_img.at(1)=Pre_point.at(1);
//   }
   ////
}
void Objection::Transform_ImgtoCam() {
    if (Real_Point.size()>0)
       Enable= true;
    else Enable= false;
}
float Objection::Get_Area_Depth(cv::Rect Box) {
    std::array<int,Stride*Stride> Arr_Box;
    int result;
    for (int i = Box.y; i < Box.y+Box.height; ++i)
        for (int j = Box.x; j < Box.x+Box.width; ++j)
        {
            if (Depthmate.at<uint16_t>(i, j) > 6000 || Depthmate.at<uint16_t>(i, j) < 200)//D435有效探测距离有限 0.2M-6M
                Arr_Box.at((i - Box.y)*Stride + (j - Box.x))= 0;
            else
                Arr_Box.at((i - Box.y)*Stride + (j - Box.x)) = Depthmate.at<uint16_t>(i, j);
        }
    sort(Arr_Box.begin(),Arr_Box.end());
    for (auto i:Arr_Box){    //最小池化
        if (i>200){
            result=i;
            break;
        }
    }
    /////////////////////////////////////
    return result  ;
}
void Objection::DealRect() {
    ///3D点聚类算法
    ///目标区域稀疏化 获取稀疏化之后的点云信息
    int height=0,width=0;
    cv::Mat Object_Area_Depth = Depthmate(Area_Objection_D);//截取目标范围的深度图
//    Mat Object_Sparse_Depth=Mat::zeros(Size(Object_Area_Depth.cols/Stride,Object_Area_Depth.rows/Stride),CV_16U);
    for (int i = 0; i < Object_Area_Depth.rows-Stride; i += Stride) {
        height++;width=0;
        for (int j = 0; j < Object_Area_Depth.cols-Stride; j += Stride) {
            array<int, 2> Sparse_Point{(Area_Objection_D.x + j + Stride / 2)>(WidthCam-1) ? (WidthCam-1):(Area_Objection_D.x + j + Stride / 2), (Area_Objection_D.y + i + Stride / 2)>(HeightCam-1) ? (HeightCam-1):Area_Objection_D.y + i + Stride / 2};
            cv::Rect Area_ele(Area_Objection_D.x + j, Area_Objection_D.y + i, Stride, Stride);
//            Object_Sparse_Depth.at<uint16_t>(i/Stride,j/Stride)=Get_Area_Depth(Area_ele);
            auto Depth_value = Get_Area_Depth(Area_ele);//稀疏化
//            if (Depth_value==0)
//                cout<<Depth_value<<endl;
//            cout<<"Position"<<Sparse_Point.at(1)<<" "<<Sparse_Point.at(0)<<endl;//测试
            Depthmate.at<uint16_t>(Sparse_Point.at(1), Sparse_Point.at(0)) = Depth_value;
//            if (Depth_value > 0) {
            Objection_DepthPoint.push_back(Sparse_Point);
            auto IP=Position_Transform(Sparse_Point, false).Report_PCL();
            Objection_PCL.push_back(IP);
            width++;
        }
    }
//    int Long=Objection_PCL.size();
//    cout<<"Test:"<<Long<<":"<<height*width<<endl;
clock_t StartTime,EndTime;
    StartTime=clock();
    Algorithm_Objection_3D This_Algorithm(Objection_PCL,height,width);
    EndTime=clock();
    auto Time=(double)(EndTime - StartTime) / CLOCKS_PER_SEC;
    Point_Camera=This_Algorithm.Center_Point;
    Real_Point=This_Algorithm.Objection_3D;

}