# Awesome Person Re-Identification

If you have any problems, suggestions or improvements, please submit the issue or PR.


## Contents
* [Datasets](#datasets)
* [Papers](#papers)
* [Leaderboard](#leaderboard)


### Code
- [Pedestrian Alignment Network](https://github.com/layumi/Pedestrian_Alignment) ![GitHub stars](https://img.shields.io/github/stars/layumi/Pedestrian_Alignment.svg?style=flat&label=Star)
- [2stream Person re-ID](https://github.com/layumi/2016_person_re-ID) ![GitHub stars](https://img.shields.io/github/stars/layumi/2016_person_re-ID.svg?style=flat&label=Star)
- [Pedestrian GAN](https://github.com/layumi/Person-reID_GAN) ![GitHub stars](https://img.shields.io/github/stars/layumi/Person-reID_GAN.svg?style=flat&label=Star)
- [Language Person Search](https://github.com/layumi/Image-Text-Embedding) ![GitHub stars](https://img.shields.io/github/stars/layumi/Image-Text-Embedding.svg?style=flat&label=Star)
- [DG-Net](https://github.com/NVlabs/DG-Net) ![GitHub stars](https://img.shields.io/github/stars/NVlabs/DG-Net.svg?style=flat&label=Star)
- [3D Person re-ID](https://github.com/layumi/person-reid-3d) ![GitHub stars](https://img.shields.io/github/stars/layumi/person-reid-3d.svg?style=flat&label=Star)
- [[L1aoXingyu](https://github.com/layumi/Person_reID_baseline_pytorch)] SOTA ReID Baseline
- [[michuanhaohao](https://github.com/michuanhaohao/reid-strong-baseline)] Bag of Tricks and A Strong Baseline for Deep Person Re-identification
- [[layumi](https://github.com/michuanhaohao/reid-strong-baseline)] A tiny, friendly, strong pytorch implement of person re-identification baseline

<!-- ### Technical blog
- [2019.05] [Chinese Blog] C^3 Framework系列之一：一个基于PyTorch的开源人群计数框架 [[Link](https://zhuanlan.zhihu.com/p/65650998)]
- [2019.04] Crowd counting from scratch [[Link](https://github.com/CommissarMa/Crowd_counting_from_scratch)]
- [2017.11] Counting Crowds and Lines with AI [[Link1](https://blog.dimroc.com/2017/11/19/counting-crowds-and-lines/)] [[Link2](https://count.dimroc.com/)] [[Code](https://github.com/dimroc/count)] -->

<!-- ###  GT generation
- Density Map Generation from Key Points [[Matlab Code](https://github.com/aachenhang/crowdcount-mcnn/tree/master/data_preparation)] [[Python Code](https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/make_dataset.ipynb)] [[Fast Python Code](https://github.com/vlad3996/computing-density-maps)] -->


## Datasets

| Dataset                   | Release time     | # identities | # cameras   | # images |
|---------------------------|------------------|--------------|-------------|----------|
| [VIPeR](https://vision.soe.ucsc.edu/node/178) | 2007 | 632 | 2 | 1264 |
| [ETH1,2,3](http://homepages.dcc.ufmg.br/~william/datasets.html)| 2007 | 85, 35, 28 | 1 | 8580 |
| [QMUL iLIDS](http://www.eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz)                | 2009             | 119          | 2           | 476      |
| [GRID](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html)                      | 2009             | 1025         | 8           | 1275     |
| [CAVIAR4ReID](http://www.lorisbazzani.info/caviar4reid.html)               | 2011             | 72           | 2           | 1220     |
| [3DPeS](http://www.openvisor.org/3dpes.asp)                     | 2011             | 192          | 8           | 1011     |
| [PRID2011](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/)                  | 2011             | 934          | 2           | 24541    |
| [V47](https://docs.google.com/leaf?id=0B692grTpU3UNZWZlN2I2NWYtYzdhNi00MWJkLWI0YjYtNTg2Zjk1OGFkMGQ1)                       | 2011             | 47           | 2           | 752      |
| [WARD](https://github.com/iN1k1/CVPR2012/tree/master/toolbox/Datasets)                      | 2012             | 70           | 3           | 4786     |
| [SAIVT-Softbio](https://researchdatafinder.qut.edu.au/display/n27416)             | 2012             | 152          | 8           | 64472    |                        |
| [CUHK01](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)                    | 2012             | 971          | 2           | 3884     |
| [CUHK02](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)                    | 2013             | 1816         | 10(5 pairs) | 7264     |
| [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)                    | 2014             | 1467         | 10(5 pairs) | 13164    |
| [RAiD](http://cs-people.bu.edu/dasabir/raid.php)                      | 2014             | 43           | 4           | 6920     |
| [iLIDS-VID](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html)                 | 2014             | 300          | 2           | 42495    |
| [MPR Drone](http://www.eecs.qmul.ac.uk/~rlayne/downloads_qmul_drone_dataset.html)                 | 2014             | 84           | 1           |          |
| [HDA Person Dataset](http://vislab.isr.ist.utl.pt/hda-dataset/)        | 2014             | 53           | 13          | 2976     |                   |
| [Shinpuhkan Dataset](http://www.mm.media.kyoto-u.ac.jp/en/datasets/shinpuhkan/)        | 2014             | 24           | 16          |          |                      |
| [CASIA Gait Database B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)     | 2015 | 124          | 11          |          |
| [Market1501](http://www.liangzheng.org/Project/project_reid.html)                | 2015             | 1501         | 6           | 32217    |
| [PKU-Reid](https://github.com/charliememory/PKU-Reid-Dataset)                  | 2016             | 114          | 2           | 1824     |
| [PRW](http://www.liangzheng.com.cn/Project/project_prw.html)                       | 2016             | 932          | 6           | 34,304    |
| [CUHK-SYSU](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html) | 2016             | 8,432       | -           | 18,184    |                    |
| [MARS](http://www.liangzheng.com.cn/Project/project_mars.html)                      | 2016             | 1261         | 6           | 1,191,003  |
| [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_baseline) (offline)             | 2017             | 1812         | 8           | 36441    |                       |
| [Airport](http://www.northeastern.edu/alert/transitioning-technology/alert-datasets/alert-airport-re-identification-dataset/)                   | 2017             | 9651         | 6           | 39902    |
| [MSMT17](http://www.pkuvmc.com/dataset.html)                    | 2018             | 4101         | 15          | 126441   |
| [RPIfield](https://drive.google.com/file/d/1GO1zm7vCAJwXgJtoFyUs367_Knz8Ev0A/view?usp=sharing)   | 2018      | 112       | 12        | 601,581       |
| [LS-VID](http://www.pkuvmc.com/dataset.html)   | 2019      | 3,772       | 15        | 2,982,685       |
| [PersonX](https://github.com/sxzrt/Dissecting-Person-Re-ID-from-the-Viewpoint-of-Viewpoint)   | 2019      | 1,266       | 6        | 273,456       |
| [COCAS](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_COCAS_A_Large-Scale_Clothes_Changing_Person_Dataset_for_Re-Identification_CVPR_2020_paper.pdf)   | 2020      | 5,266       | -        | 62,382       |

## Papers



### Survey
- <a name=""></a> Person search: New paradigm of person re-identification: A survey and outlook of recent works (**IVC2020**) [[paper](https://www.sciencedirect.com/science/article/pii/S0262885620301025)]
- <a name=""></a> Deep Learning for Person Re-identification: A Survey and Outlook (**arXiv**) [[arxiv](https://arxiv.org/abs/2001.04193)]
- <a name=""></a> A Survey of Open-World Person Re-identification (**T-CSVT2019**) [[paper](https://ieeexplore.ieee.org/abstract/document/8640834)]
- <a name=""></a> A Systematic Evaluation and Benchmark for Person Re-Identification: Features, Metrics, and Datasets **(T-PAMI2018)** [[paper](https://ieeexplore.ieee.org/document/8294254/)][[github](https://github.com/RSL-NEU/person-reid-benchmark)] [[arxiv](https://arxiv.org/abs/1605.09653)]
- <a name=""></a> Person Re-identification: Past, Present and Future (**arXiv2016**) [[arxiv](https://arxiv.org/abs/1610.02984)]
- <a name=""></a> A survey of approaches and trends in person re-identification (**Image and Vision Computing 2014**) [[paper](https://www.sciencedirect.com/science/article/pii/S0262885614000262)]
- <a name=""></a> Appearance Descriptors for Person Re-identification: a Comprehensive Review (**arXiv2013**) [[arxiv](https://arxiv.org/abs/1307.574)]
- <a name=""></a> People reidentification in surveillance and forensics: A survey (**CSUR2013**) [[paper](https://dl.acm.org/citation.cfm?id=2543596)]
- <a name=""></a> Intelligent multi-camera video surveillance: A review (**PR Letters2013**) [[paper](https://www.sciencedirect.com/science/article/pii/S016786551200219X)]

### Methods dealing with the lack of labelled data
- <a name=""></a> Unsupervised Person Re-Identification via Softened Similarity Learning
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Unsupervised_Person_Re-Identification_via_Softened_Similarity_Learning_CVPR_2020_paper.pdf)]
- <a name=""></a> Unsupervised Person Re-identification via Multi-label Classification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf)]
- <a name=""></a>  Asymetric Co-Teaching for Unsupervised Cross Domain Person Re-Identification **(AAAI2020)** [[arxiv](https://arxiv.org/abs/1912.01349)]
- <a name=""></a> Domain Adaptive Attention Learning for Unsupervised Person Re-Identification **(AAAI2020)** 
- <a name=""></a> Tracklet Self-Supervised Learning for Unsupervised Person Re-Identification **(AAAI2020)** [[paper](http://www.eecs.qmul.ac.uk/~sgg/papers/WuEtAl_AAAI2020.pdf)]
- <a name="SHRED"></a> **[SHRED]** Unsupervised Domain Adaptation in Person re-ID via k-Reciprocal Clustering and Large-Scale Heterogeneous Environment Synthesis **(WACV2020)** [[paper](http://openaccess.thecvf.com/content_WACV_2020/html/Kumar_Unsupervised_Domain_Adaptation_in_Person_re-ID_via_k-Reciprocal_Clustering_and_WACV_2020_paper.html)]
- <a name="CamStyle"></a> **[CamStyle]** CamStyle: A Novel Data Augmentation Method for Person Re-Identification (**TIP2019**) [[paper](https://ieeexplore.ieee.org/document/8485427/)][[github](https://github.com/zhunzhong07/CamStyle)]
- <a name="DGM+"></a> **[DGM+]** Dynamic Graph Co-Matching for Unsupervised Video-Based Person Re-Identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8611378)]
- <a name="t-MTL"></a> **[t-MTL]** Tensor Multi-task Learning for Person Re-identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8889995)]
- <a name="UTAL"></a> **[UTAL]** Unsupervised Tracklet Person Re-Identification (**T-PAMI2019**) [[paper](https://ieeexplore.ieee.org/abstract/document/8658110)][[github](https://github.com/liminxian/DukeMTMC-SI-Tracklet)]
- <a name="MAR"></a> **[MAR]** Unsupervised Person Re-identification by Soft Multilabel Learning (**CVPR2019**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Unsupervised_Person_Re-Identification_by_Soft_Multilabel_Learning_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1903.06325.pdf)] [[github](https://github.com/KovenYu/MAR)]
- <a name="E2E"></a> **[E2E]** Unsupervised Person Image Generation with Semantic Parsing Transformation (**CVPR2019**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Song_Unsupervised_Person_Image_Generation_With_Semantic_Parsing_Transformation_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1904.03379.pdf)] [[github](https://github.com/SijieSong/person_generation_spt)]
- <a name="PAUL"></a> **[PAUL]** Patch-Based Discriminative Feature Learning for Unsupervised Person Re-Identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Patch-Based_Discriminative_Feature_Learning_for_Unsupervised_Person_Re-Identification_CVPR_2019_paper.html)]
- <a name="BUC"></a> **[BUC]** A Bottom-Up Clustering Approach to Unsupervised Person Re-identification (**AAAI2019**) (**Oral**) [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/4898)] [[github](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification)]
- <a name=""></a> Weakly Supervised Person Re-Identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Meng_Weakly_Supervised_Person_Re-Identification_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1904.03832.pdf)]
- <a name=""></a> Distilled Person Re-identification: Towards a More Scalable System (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Distilled_Person_Re-Identification_Towards_a_More_Scalable_System_CVPR_2019_paper.html)]
- <a name="SSG++"></a> **[SSG++]** Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification (**ICCV2019**) (**Oral**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Fu_Self-Similarity_Grouping_A_Simple_Unsupervised_Cross_Domain_Adaptation_Approach_for_ICCV_2019_paper.html)][[arxiv](https://arxiv.org/abs/1811.10144)]
- <a name="UCDA-CCE"></a> **[UCDA-CCE]** A Novel Unsupervised Camera-Aware Domain Adaptation Framework for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Qi_A_Novel_Unsupervised_Camera-Aware_Domain_Adaptation_Framework_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1904.03425)]
- <a name=""></a> Self-Training With Progressive Augmentation for Unsupervised Cross-Domain Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Self-Training_With_Progressive_Augmentation_for_Unsupervised_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1907.13315)]
- <a name="UGA"></a> **[UGA]** Unsupervised Graph Association for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Unsupervised_Graph_Association_for_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name="CFCL"></a> **[CFCL]** Unsupervised Person Re-Identification by Camera-Aware Similarity Consistency Learning (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Unsupervised_Person_Re-Identification_by_Camera-Aware_Similarity_Consistency_Learning_ICCV_2019_paper.html)]
- <a name="PDA-Net"></a> **[PDA-Net]** Cross-Dataset Person Re-Identification via Unsupervised Pose Disentanglement and Adaptation (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Li_Cross-Dataset_Person_Re-Identification_via_Unsupervised_Pose_Disentanglement_and_Adaptation_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1909.09675)]
- <a name=""></a> Deep Reinforcement Active Learning for Human-in-the-Loop Person Re-Identification (**ICCV2019**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Deep_Reinforcement_Active_Learning_for_Human-in-the-Loop_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name="TJ-AIDL"></a> **[TJ-AIDL]** Transferable Joint Attribute-Identity Deep Learning for Unsupervised Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Transferable_Joint_Attribute-Identity_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1803.09786)]
- <a name=""></a> Unsupervised Cross-Dataset Person Re-Identification by Transfer Learning of Spatial-Temporal Patterns (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Lv_Unsupervised_Cross-Dataset_Person_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1803.07293)]
- <a name="DAsy"></a> **[DAsy]** Domain Adaptation through Synthesis for Unsupervised Person Re-identification (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper.html)]
- <a name="RACE"></a> **[RACE]** Robust Anchor Embedding for Unsupervised Video Person Re-Identification in the Wild (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Mang_YE_Robust_Anchor_Embedding_ECCV_2018_paper.html)]
- <a name="TAUDL"></a> **[TAUDL]** Unsupervised Person Re-identification by Deep Learning Tracklet Association (**ECCV2018**)(Oral) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Minxian_Li_Unsupervised_Person_Re-identification_ECCV_2018_paper.html)]
- <a name="CAMEL"></a> **[CAMEL]** Cross-View Asymmetric Metric Learning for Unsupervised Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Yu_Cross-View_Asymmetric_Metric_ICCV_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1708.08062)]
- <a name=""></a> Stepwise Metric Promotion for Unsupervised Video Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Stepwise_Metric_Promotion_ICCV_2017_paper.html)]
- <a name="UDAP"></a> **[UDAP]** Unsupervised Domain Adaptive Re-Identification: Theory and Practice (**arXiv2018**) [[paper](https://arxiv.org/abs/1807.11334)]
- <a name="ECN"></a> **[ECN]** Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification (**CVPR2019**) [[paper](https://arxiv.org/abs/1904.01990)]
- <a name="CDs"></a> **[CDs]** Clustering and Dynamic Sampling Based Unsupervised Domain Adaptation for Person Re-Identification (**ICME2019**) [[paper](http://www.cbsr.ia.ac.cn/users/zlei/papers/JLWU-ICME-2019.pdf)]
- <a name="ARN"></a> **[ARN]** Adaptation and reidentification network: An unsupervised deep transfer learning approach to person re-identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Li_Adaptation_and_Re-Identification_CVPR_2018_paper.pdf)]
- <a name="HHL"></a> **[HHL]** Generalizing a person retrieval model hetero-and homogeneously (**ECCV2018**) [[paper](https://uploads-ssl.webflow.com/5cd23e823ab9b1f01f815a54/5d10d8990022bdb066a9491d_Generalizing%20A%20Person%20Retrieval%20Model%20Hetero%20and%20Homogeneously.pdf)]
- <a name="DECAMEL"></a> **[DECAMEL]** Unsupervised person re- identification by deep asymmetric metric embedding (**PAMI2019**) [[paper](https://arxiv.org/pdf/1901.10177.pdf)]
- <a name="SPGAN+LMP"></a> **[SPGAN+LMP]** Image-image domain adaptation with preserved self-similarity and domain-dissimilarity for person re-identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.pdf)]
- <a name="MMFA"></a> **[MMFA]** Multi-task mid-level feature alignment network for un- supervised cross-dataset person re-identification (**BMVC2018**) [[paper](http://www.bmva.org/bmvc/2018/contents/papers/0244.pdf)]
- <a name="CycleGAN"></a> **[CycleGAN]** Unpaired image-to- image translation using cycle-consistent adversarial networks (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)]
- <a name="PUL"></a> **[PUL]**  Unsupervised person re-identification: Clustering and fine-tuning (**TOMM2018**) [[paper](https://arxiv.org/pdf/1705.10444.pdf)]
- <a name="PTGAN"></a> **[PTGAN]** Person transfer gan to bridge domain gap for person re-identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Person_Transfer_GAN_CVPR_2018_paper.pdf)]
- <a name="proposed"></a> **[proposed]** Video-Based Person Re-Identification Using Unsupervised Tracklet Matching (**Access2019**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8639924)]
- <a name="DAL"></a> **[DAL]** Deep association learning for unsupervised video person re-identification (**BMVC2018**) [[paper](https://arxiv.org/pdf/1808.07301.pdf)]
- <a name="SMP*"></a> **[SMP*]** Stepwise metric promotion for unsupervised video person re-identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Stepwise_Metric_Promotion_ICCV_2017_paper.pdf)]
- <a name="PAM+LOMO"></a> **[PAM+LOMO]** Multi-shot person re-identification using part appearance mixture (**WACV**) [[paper](http://www-sop.inria.fr/members/Francois.Bremond/Postscript/SalwaSSD18.pdf)]
- <a name="DGM+IDE"></a> **[DGM+IDE]** Dynamic label graph matching for unsupervised video re-identification (**ICCV2017**) [[paper](https://arxiv.org/abs/1709.09297v1)]
- <a name="MDTS"></a> **[MDTS]** Person re-identification by unsupervised video matching (**PR2017**) [[paper](https://arxiv.org/abs/1611.08512)]

### 2020
- <a name=""></a> COCAS: A Large-Scale Clothes Changing Person Dataset for Re-Identification **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_COCAS_A_Large-Scale_Clothes_Changing_Person_Dataset_for_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> Online Joint Multi-Metric Adaptation From Frequent Sharing-Subset Mining for Person Re-Identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Online_Joint_Multi-Metric_Adaptation_From_Frequent_Sharing-Subset_Mining_for_Person_CVPR_2020_paper.pdf)]
- <a name=""></a> Style Normalization and Restitution for Generalizable Person Re-Identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Style_Normalization_and_Restitution_for_Generalizable_Person_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> Unsupervised Person Re-Identification via Softened Similarity Learning
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Unsupervised_Person_Re-Identification_via_Softened_Similarity_Learning_CVPR_2020_paper.pdf)]
- <a name=""></a> Transferable, Controllable, and Inconspicuous Adversarial Attacks on Person Re-identification With Deep Mis-Ranking
 **(CVPR2020)(Oral)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Transferable_Controllable_and_Inconspicuous_Adversarial_Attacks_on_Person_Re-identification_With_CVPR_2020_paper.pdf)]
- <a name=""></a> Inter-Task Association Critic for Cross-Resolution Person Re-Identification
 **(CVPR2020)(Oral)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Inter-Task_Association_Critic_for_Cross-Resolution_Person_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> Learning Multi-Granular Hypergraphs for Video-Based Person Re-Identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yan_Learning_Multi-Granular_Hypergraphs_for_Video-Based_Person_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> Relation-Aware Global Attention for Person Re-identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Relation-Aware_Global_Attention_for_Person_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> Spatial-Temporal Graph Convolutional Network for Video-based
Person Re-identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Spatial-Temporal_Graph_Convolutional_Network_for_Video-Based_Person_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> Salience-Guided Cascaded Suppression Network for Person Re-identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Salience-Guided_Cascaded_Suppression_Network_for_Person_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> Bi-directional Interaction Network for Person Search
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_Bi-Directional_Interaction_Network_for_Person_Search_CVPR_2020_paper.pdf)]
- <a name=""></a> Instance Guided Proposal Network for Person Search
 **(CVPR2020)(Oral)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_Instance_Guided_Proposal_Network_for_Person_Search_CVPR_2020_paper.pdf)]
- <a name=""></a> AD-Cluster: Augmented Discriminative Clustering for Domain Adaptive
Person Re-identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhai_AD-Cluster_Augmented_Discriminative_Clustering_for_Domain_Adaptive_Person_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> Unity Style Transfer for Person Re-Identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Unity_Style_Transfer_for_Person_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> High-Order Information Matters: Learning Relation and Topology
for Occluded Person Re-Identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_High-Order_Information_Matters_Learning_Relation_and_Topology_for_Occluded_Person_CVPR_2020_paper.pdf)]
- <a name=""></a> Robust Partial Matching for Person Search in the Wild
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhong_Robust_Partial_Matching_for_Person_Search_in_the_Wild_CVPR_2020_paper.pdf)]
- <a name=""></a> Weakly Supervised Discriminative Feature Learning with State Information
for Person Identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Weakly_Supervised_Discriminative_Feature_Learning_With_State_Information_for_Person_CVPR_2020_paper.pdf)]
- <a name=""></a> Hi-CMD: Hierarchical Cross-Modality Disentanglement for Visible-Infrared
Person Re-Identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Choi_Hi-CMD_Hierarchical_Cross-Modality_Disentanglement_for_Visible-Infrared_Person_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> Learning Longterm Representations for Person Re-Identification
Using Radio Signals
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Learning_Longterm_Representations_for_Person_Re-Identification_Using_Radio_Signals_CVPR_2020_paper.pdf)]
- <a name=""></a> Camera On-boarding for Person Re-identification using Hypothesis Transfer
Learning
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ahmed_Camera_On-Boarding_for_Person_Re-Identification_Using_Hypothesis_Transfer_Learning_CVPR_2020_paper.pdf)]
- <a name=""></a> Cross-modality Person re-identification with Shared-Specific Feature Transfer
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_Cross-Modality_Person_Re-Identification_With_Shared-Specific_Feature_Transfer_CVPR_2020_paper.pdf)]
- <a name=""></a> Hierarchical Clustering with Hard-batch Triplet Loss for Person
Re-identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_Hierarchical_Clustering_With_Hard-Batch_Triplet_Loss_for_Person_Re-Identification_CVPR_2020_paper.pdf)]
- <a name=""></a> Real-world Person Re-Identification via Degradation Invariance Learning
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Real-World_Person_Re-Identification_via_Degradation_Invariance_Learning_CVPR_2020_paper.pdf)]
- <a name=""></a> Unsupervised Person Re-identification via Multi-label Classification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf)]
- <a name=""></a> Smoothing Adversarial Domain Attack and p-Memory Reconsolidation for
Cross-Domain Person Re-Identification
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Smoothing_Adversarial_Domain_Attack_and_P-Memory_Reconsolidation_for_Cross-Domain_Person_CVPR_2020_paper.pdf)]
- <a name=""></a> Norm-Aware Embedding for Efficient Person Search
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Norm-Aware_Embedding_for_Efficient_Person_Search_CVPR_2020_paper.pdf)]
- <a name=""></a> TCTS: A Task-Consistent Two-stage Framework for Person Search
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_TCTS_A_Task-Consistent_Two-Stage_Framework_for_Person_Search_CVPR_2020_paper.pdf)]
- <a name=""></a> Pose-guided Visible Part Matching for Occluded Person ReID
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gao_Pose-Guided_Visible_Part_Matching_for_Occluded_Person_ReID_CVPR_2020_paper.pdf)]
- <a name=""></a> Cross-Modal Cross-Domain Moment Alignment Network for Person Search
 **(CVPR2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jing_Cross-Modal_Cross-Domain_Moment_Alignment_Network_for_Person_Search_CVPR_2020_paper.pdf)]
- <a name=""></a> Uncertainty-aware Multi-shot Knowledge Distillation for Image-based Object Re-identification **(AAAI2020)** [[arxiv](https://arxiv.org/abs/2001.05197)]
- <a name=""></a> Infrared-Visible Cross-Modal Person Re-Identification with an X Modality **(AAAI2020)** 
- <a name=""></a>  Single Camera Training for Person Re-identification **(AAAI2020)** [[arxiv](https://arxiv.org/abs/1909.10848)]
- <a name=""></a>  Cross-Modality Paired-Images Generation for RGB-Infrared Person Re-Identification **(AAAI2020)** [[arxiv](https://arxiv.org/abs/2002.04114)]
- <a name=""></a>  Relation-Guided Spatial Attention and Temporal Refinement for Video-based Person Re-Identification **(AAAI2020)** 
- <a name=""></a>  Semantics-Aligned Representation Learning for Person Re-identification **(AAAI2020)** [[arxiv](https://arxiv.org/abs/1905.13143)]
- <a name=""></a>  Relation Network for Person Re-identification **(AAAI2020)** [[arxiv](https://arxiv.org/abs/1911.09318)]
- <a name=""></a> Appearance and Motion Enhancement for Video-based Person Re-identification **(AAAI2020)** 
- <a name=""></a>  Rethinking Temporal Fusion for Video-based Person Re-identification on Semantic and Time Aspect **(AAAI2020)** [[arxiv](https://arxiv.org/abs/1911.12512)]
- <a name=""></a> Frame-Guided Region-Aligned Representation for Video Person Re-identification **(AAAI2020)** 
- <a name=""></a> Viewpoint-Aware Loss with Angular Regularization for Person Re-Identification **(AAAI2020)** 
- <a name=""></a> Semantic Consistency and Identity Mapping Multi-Component Generative Adversarial Network for Person Re-Identification **(WACV2020)** [[paper](http://openaccess.thecvf.com/content_WACV_2020/html/Khatun_Semantic_Consistency_and_Identity_Mapping_Multi-Component_Generative_Adversarial_Network_for_WACV_2020_paper.html)]
- <a name="SCR"></a> **[SCR]** Learning Discriminative and Generalizable Representations by Spatial-Channel Partition for Person Re-Identification **(WACV2020)** [[paper](http://openaccess.thecvf.com/content_WACV_2020/html/Chen_Learning_Discriminative_and_Generalizable_Representations_by_Spatial-Channel_Partition_for_Person_WACV_2020_paper.html)]
- <a name=""></a> Video Person Re-Identification using Learned Clip Similarity Aggregation **(WACV2020)** [[paper](http://openaccess.thecvf.com/content_WACV_2020/html/Matiyali_Video_Person_Re-Identification_using_Learned_Clip_Similarity_Aggregation_WACV_2020_paper.html)]
- <a name=""></a> Pose Guided Gated Fusion for Person Re-identification **(WACV2020)** [[paper](http://openaccess.thecvf.com/content_WACV_2020/html/Bhuiyan_Pose_Guided_Gated_Fusion_for_Person_Re-identification_WACV_2020_paper.html)]
- <a name=""></a> Temporal Aggregation with Clip-level Attention for Video-based Person Re-identification **(WACV2020)** [[paper](http://openaccess.thecvf.com/content_WACV_2020/html/Li_Temporal_Aggregation_with_Clip-level_Attention_for_Video-based_Person_Re-identification_WACV_2020_paper.html)]
- <a name=""></a> Calibrated Domain-Invariant Learning for Highly Generalizable Large Scale Re-Identification **(WACV2020)** [[paper](http://openaccess.thecvf.com/content_WACV_2020/html/Yuan_Calibrated_Domain-Invariant_Learning_for_Highly_Generalizable_Large_Scale_Re-Identification_WACV_2020_paper.html)]


### 2019
- <a name="REGCT"></a> **[REGCT]** Robust and Efficient Graph Correspondence Transfer for Person Re-identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8709994)]
- <a name="DHA"></a> **[DHA]** Learning Sparse and Identity-preserved Hidden Attributes for Person Re-identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8874954)]
- <a name="RAP"></a> **[RAP]** A Richly Annotated Pedestrian Dataset for Person Retrieval in Real Surveillance Scenarios **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8510891)][[github](https://github.com/dangweili/RAP)]
- <a name="SCAN"></a> **[SCAN]** SCAN: Self-and-Collaborative Attention Network for Video Person Re-Identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8703416)][[github](https://github.com/ruixuejianfei/SCAN)]
- <a name="FANN"></a> **[FANN]** Discriminative Feature Learning With Foreground Attention for Person Re-Identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8676064)]
- <a name=""></a> Progressive Learning for Person Re-Identification With One Example **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8607049)][[github](https://github.com/Yu-Wu/One-Example-Person-ReID)]
- <a name="PIE"></a> **[PIE]** Pose-Invariant Embedding for Deep Person Re-Identification **(TIP2019)** [[paper](hhttps://ieeexplore.ieee.org/document/8693885)]
- <a name="UVDL"></a> **[UVDL]** Uniform and Variational Deep Learning for RGB-D Object Recognition and Person Re-Identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8715446)]
- <a name="CI-CNN"></a> **[CI-CNN]** Context-Interactive CNN for Person Re-Identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8907836)]
- <a name="k-KISSME"></a> **[k-KISSME]** Kernel Distance Metric Learning Using Pairwise Constraints for Person Re-Identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8469088)][[github](https://github.com/bacnguyencong/k-KISSME)]
- <a name="MpRL"></a> **[MpRL]** Multi-Pseudo Regularized Label for Generated Data in Person Re-Identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8485730)][[github](https://github.com/Huang-3/MpRL-for-person-re-ID)]
- <a name=""></a> Deep Representation Learning With Part Loss for Person Re-Identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8607050)]
- <a name="TRL"></a> **[TRL]** Video Person Re-Identification by Temporal Residual Learning **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8513884)]
- <a name="STAL"></a> **[STAL]** Spatial-Temporal Attention-Aware Learning for Video-Based Person Re-Identification **(TIP2019)** [[paper](https://ieeexplore.ieee.org/document/8675957)]
- <a name="MuDeep"></a> **[MuDeep]** Leader-based Multi-Scale Attention Deep Architecture for Person Re-identification **(T-PAMI2019)** [[paper](https://ieeexplore.ieee.org/document/8762210)]
- <a name=""></a> Learning Part-based Convolutional Features for Person Re-identification **(T-PAMI2019)** [[paper](https://ieeexplore.ieee.org/document/8826008)]
- <a name="PGR"></a> **[PGR]** Pose-Guided Representation Learning for Person Re-Identification **(T-PAMI2019)** [[paper](https://ieeexplore.ieee.org/document/8764426)]
- <a name="HGD"></a> **[HGD]** Hierarchical Gaussian Descriptors with Application to Person Re-Identification **(T-PAMI2019)** [[paper](https://ieeexplore.ieee.org/document/8705270)]
- <a name=""></a> A Graph-based Approach for Making Consensus-based Decisions in Image Search and Person Re-identification **(T-PAMI2019)** [[paper](https://ieeexplore.ieee.org/document/8852741)]
- <a name="KPM|RW"></a> **[KPM|RW]** Person Re-identification with Deep Kronecker-Product Matching and Group-shuffling Random Walk **(T-PAMI2019)** [[paper](https://ieeexplore.ieee.org/document/8906139)]
- <a name="MHP"></a> **[MHP]** Fine-Grained Multi-human Parsing (**IJCV2019**) [[paper](https://link.springer.com/article/10.1007/s11263-019-01181-5)]
- <a name="TPI"></a> **[TPI]**  Tracking Persons-of-Interest via Unsupervised Representation Adaptation (**IJCV2019**) [[paper](https://link.springer.com/article/10.1007/s11263-019-01212-1)]
- <a name="FCDSC"></a> **[FCDSC]** Multi-target Tracking in Multiple Non-overlapping Cameras Using Fast-Constrained Dominant Sets **(IJCV2019)** [[paper](https://link.springer.com/article/10.1007/s11263-019-01180-6)]
- <a name="DAN"></a> **[DAN]** Learning Discriminative Aggregation Network for Video-Based Face Recognition and Person Re-identification (**IJCV2019**) [[paper](https://link.springer.com/article/10.1007/s11263-018-1135-x)]
- <a name=""></a>  Learning Disentangled Representation for Robust Person Re-identification (**NeurIPS2019**) [[paper](https://papers.nips.cc/paper/8771-learning-disentangled-representation-for-robust-person-re-identification)]
- <a name="AlignedReID++"></a>  **[AlignedReID++]**  AlignedReID++: Dynamically matching local information for person re-identification (**PR2019**) [[paper](https://www.sciencedirect.com/science/article/pii/S0031320319302031)]
- <a name="MSP-CNN"></a> **[MSP-CNN]** Multi-level Similarity Perception Network for Person Re-identification (**TOMM2019**) [[paper](https://dl.acm.org/citation.cfm?id=3309881)]
- <a name=""></a> Discriminative Representation Learning for Person Re-identification via Multi-loss Training (**JVCIR2019**) [[paper](https://www.sciencedirect.com/science/article/pii/S1047320319301749)]
- <a name="DG-Net"></a> **[DG-Net]** Joint Discriminative and Generative Learning for Person Re-identification (**CVPR2019**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Joint_Discriminative_and_Generative_Learning_for_Person_Re-Identification_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1904.07223.pdf)]
- <a name="DSA-reID"></a> **[DSA-reID]** Densely Semantically Aligned Person Re-Identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Densely_Semantically_Aligned_Person_Re-Identification_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1812.08967.pdf)]
- <a name="DIMN"></a> **[DIMN]** Generalizable Person Re-identification by Domain-Invariant Mapping Network (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Song_Generalizable_Person_Re-Identification_by_Domain-Invariant_Mapping_Network_CVPR_2019_paper.html)]
- <a name="CASN"></a> **[CASN]** Re-Identification with Consistent Attentive Siamese Networks (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Re-Identification_With_Consistent_Attentive_Siamese_Networks_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1811.07487.pdf)]
- <a name=""></a> Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Invariance_Matters_Exemplar_Memory_for_Domain_Adaptive_Person_Re-Identification_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1904.01990.pdf)] [[github](https://github.com/zhunzhong07/ECN)]
- <a name=""></a> Re-ranking via Metric Fusion for Object Retrieval and Person Re-identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Bai_Re-Ranking_via_Metric_Fusion_for_Object_Retrieval_and_Person_Re-Identification_CVPR_2019_paper.html)]
- <a name=""></a> Progressive Pose Attention Transfer for Person Image Generation (**CVPR2019**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Progressive_Pose_Attention_Transfer_for_Person_Image_Generation_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1904.03349.pdf)] [[github](https://github.com/tengteng95/Pose-Transfer)]
- <a name=""></a> Learning to Reduce Dual-level Discrepancy for Infrared-Visible Person Re-identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Learning_to_Reduce_Dual-Level_Discrepancy_for_Infrared-Visible_Person_Re-Identification_CVPR_2019_paper.html)]
- <a name=""></a> Text Guided Person Image Synthesis (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_Text_Guided_Person_Image_Synthesis_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1904.05118.pdf)]
- <a name=""></a> Learning Context Graph for Person Search (**CVPR2019**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Yan_Learning_Context_Graph_for_Person_Search_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1904.01830.pdf)] [[github](https://github.com/sjtuzq/person_search_gcn)]
- <a name="QEEPS"></a> **[QEEPS]** Query-guided End-to-End Person Search (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Munjal_Query-Guided_End-To-End_Person_Search_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1905.01203.pdf)] [[github](https://github.com/munjalbharti/Query-guided-End-to-End-Person-Search)]
- <a name=""></a> Multi-person Articulated Tracking with Spatial and Temporal Embeddings (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Jin_Multi-Person_Articulated_Tracking_With_Spatial_and_Temporal_Embeddings_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1903.09214.pdf)]
- <a name=""></a> Dissecting Person Re-identification from the Viewpoint of Viewpoint (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Dissecting_Person_Re-Identification_From_the_Viewpoint_of_Viewpoint_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1812.02162.pdf)] [[github](https://github.com/sxzrt/Dissecting-Person-Re-ID-from-the-Viewpoint-of-Viewpoint)]
- <a name="CAMA"></a> **[CAMA]** Towards Rich Feature Discovery with Class Activation Maps Augmentation for Person Re-Identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Towards_Rich_Feature_Discovery_With_Class_Activation_Maps_Augmentation_for_CVPR_2019_paper.html)]
- <a name="VRSTC"></a> **[VRSTC]** VRSTC: Occlusion-Free Video Person Re-Identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Hou_VRSTC_Occlusion-Free_Video_Person_Re-Identification_CVPR_2019_paper.html)]
- <a name="ATNet"></a> **[ATNet]** Adaptive Transfer Network for Cross-Domain Person Re-Identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Adaptive_Transfer_Network_for_Cross-Domain_Person_Re-Identification_CVPR_2019_paper.html)]
- <a name="Pyramid"></a> **[Pyramid]** Pyramidal Person Re-IDentification via Multi-Loss Dynamic Training (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Pyramidal_Person_Re-IDentification_via_Multi-Loss_Dynamic_Training_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1810.12193.pdf)]
- <a name="IANet"></a> **[IANet]** Interaction-and-Aggregation Network for Person Re-identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Interaction-And-Aggregation_Network_for_Person_Re-Identification_CVPR_2019_paper.html)]
- <a name=""></a> Skin-based identification from multispectral image data using CNNs (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Uemori_Skin-Based_Identification_From_Multispectral_Image_Data_Using_CNNs_CVPR_2019_paper.html)]
- <a name="VPM"></a> **[VPM]** Perceive Where to Focus: Learning Visibility-Aware Part-Level Features for Partial Person Re-Identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Perceive_Where_to_Focus_Learning_Visibility-Aware_Part-Level_Features_for_Partial_CVPR_2019_paper.html)] [[github](https://github.com/sxzrt/Dissecting-Person-Re-ID-from-the-Viewpoint-of-Viewpoint)]
- <a name=""></a> Attribute-Driven Feature Disentangling and Temporal Aggregation for Video Person Re-Identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Attribute-Driven_Feature_Disentangling_and_Temporal_Aggregation_for_Video_Person_Re-Identification_CVPR_2019_paper.html)]
- <a name=""></a> DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Ge_DeepFashion2_A_Versatile_Benchmark_for_Detection_Pose_Estimation_Segmentation_and_CVPR_2019_paper.html)]
- <a name="AANe"></a> **[AANe]** AANet: Attribute Attention Network for Person Re-Identifications (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Tay_AANet_Attribute_Attention_Network_for_Person_Re-Identifications_CVPR_2019_paper.html)]
- <a name=""></a> Re-Identification Supervised Texture Generation (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Re-Identification_Supervised_Texture_Generation_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1904.03385.pdf)]
- <a name="st-ReID"></a> **[st-ReID]** Spatial-Temporal Person Re-identification (**AAAI2019**) [[arxiv](https://arxiv.org/abs/1812.03282)] [[github](https://github.com/Wanggcong/Spatial-Temporal-Re-identification)]
- <a name=""></a> Learning Resolution-Invariant Deep Representations for Person Re-Identification (**AAAI2019**)(**Oral**) [[paper](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4832)]
- <a name="HSME"></a> **[HSME]** HSME: Hypersphere Manifold Embedding for Visible Thermal Person Re-identification (**AAAI2019**) [[paper](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4853)]
- <a name="HPM"></a> **[HPM]** Horizontal Pyramid Matching for Person Re-identification (**AAAI2019**) [[arxiv](https://arxiv.org/abs/1804.05275)]
- <a name="STA"></a> **[STA]** STA: Spatial-Temporal Attention for Large-Scale Video-based Person Re-Identification (**AAAI2019**) [[arxiv](https://arxiv.org/abs/1811.04129)]
- <a name=""></a> Multi-scale 3D Convolution Network for Video Based Person Re-Identification (**AAAI2019**) [[arxiv](https://arxiv.org/abs/1811.07468)]
- <a name=""></a> Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person
Re-Identification (**AAAI2019**) [[arxiv](https://arxiv.org/abs/1901.06140)] [[github](https://github.com/youngminPIL/rollback)]
- <a name=""></a> Spatial and Temporal Mutual Promotion for Video-based Person Re-identification (**AAAI2019**) [[arxiv](https://arxiv.org/abs/1812.10305)]
- <a name=""></a> Learning Incremental Triplet Margin for Person Re-identification (**AAAI2019**) [[arxiv](https://arxiv.org/abs/1812.06576)]
- <a name="KVM-MN"></a> **[KVM-MN]** Learning A Key-Value Memory Co-Attention Matching Network for Person ReIdentification (**AAAI2019**) [[paper](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4959)]
- <a name="ABD-Net"></a> **[ABD-Net]** ABD-Net: Attentive but Diverse Person Re-Identification (**ICCV2019**) [[github](https://github.com/TAMU-VITA/ABD-Net)] [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Chen_ABD-Net_Attentive_but_Diverse_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1908.01114)]
- <a name="advPattern"></a> **[advPattern]** advPattern: Physical-World Attacks on Deep Person Re-Identification via Adversarially Transformable Patterns (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Wang_advPattern_Physical-World_Attacks_on_Deep_Person_Re-Identification_via_Adversarially_Transformable_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1908.09327)]
- <a name=""></a> Instance-Guided Context Rendering for Cross-Domain Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Instance-Guided_Context_Rendering_for_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> Mixed High-Order Attention Network for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Mixed_High-Order_Attention_Network_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1908.05819)]
- <a name=""></a> Recover and Identify: A Generative Dual Model for Cross-Resolution Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Li_Recover_and_Identify_A_Generative_Dual_Model_for_Cross-Resolution_Person_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1908.06052)]
- <a name=""></a> Pose-Guided Feature Alignment for Occluded Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Miao_Pose-Guided_Feature_Alignment_for_Occluded_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> Robust Person Re-Identification by Modelling Feature Uncertainty (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Yu_Robust_Person_Re-Identification_by_Modelling_Feature_Uncertainty_ICCV_2019_paper.html)]
- <a name=""></a> Co-Segmentation Inspired Attention Networks for Video-Based Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Subramaniam_Co-Segmentation_Inspired_Attention_Networks_for_Video-Based_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> RGB-Infrared Cross-Modality Person Re-Identification via Joint Pixel and Feature Alignment (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Wang_RGB-Infrared_Cross-Modality_Person_Re-Identification_via_Joint_Pixel_and_Feature_Alignment_ICCV_2019_paper.html)]
- <a name=""></a> Beyond Human Parts: Dual Part-Aligned Representations for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Guo_Beyond_Human_Parts_Dual_Part-Aligned_Representations_for_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> Batch DropBlock Network for Person Re-Identification and Beyond (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Dai_Batch_DropBlock_Network_for_Person_Re-Identification_and_Beyond_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1811.07130)]
- <a name=""></a> Omni-Scale Feature Learning for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1905.00953)]
- <a name=""></a> Auto-ReID: Searching for a Part-Aware ConvNet for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Quan_Auto-ReID_Searching_for_a_Part-Aware_ConvNet_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1903.09776)]
- <a name=""></a> Second-Order Non-Local Attention Networks for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Xia_Second-Order_Non-Local_Attention_Networks_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1909.00295)]
- <a name=""></a> Global-Local Temporal Representations for Video Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Li_Global-Local_Temporal_Representations_for_Video_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1908.10049)] [[github](https://github.com/kanei1024/GLTR)]
- <a name=""></a> Spectral Feature Transformation for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Luo_Spectral_Feature_Transformation_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1811.11405)]
- <a name=""></a> View Confusion Feature Learning for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Liu_View_Confusion_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> MVP Matching: A Maximum-Value Perfect Matching for Mining Hard Samples, With Application to Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Sun_MVP_Matching_A_Maximum-Value_Perfect_Matching_for_Mining_Hard_Samples_ICCV_2019_paper.html)]
- <a name=""></a> Discriminative Feature Learning With Consistent Attention Regularization for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Discriminative_Feature_Learning_With_Consistent_Attention_Regularization_for_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> Foreground-Aware Pyramid Reconstruction for Alignment-Free Occluded Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/He_Foreground-Aware_Pyramid_Reconstruction_for_Alignment-Free_Occluded_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1904.04975)]
- <a name=""></a> SBSGAN: Suppression of Inter-Domain Background Shift for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Huang_SBSGAN_Suppression_of_Inter-Domain_Background_Shift_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1908.09086)]
- <a name=""></a> Self-Critical Attention Learning for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Self-Critical_Attention_Learning_for_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> Temporal Knowledge Propagation for Image-to-Video Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Gu_Temporal_Knowledge_Propagation_for_Image-to-Video_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1908.03885)]
- <a name=""></a> Deep Constrained Dominant Sets for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Alemu_Deep_Constrained_Dominant_Sets_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1904.11397)]



### 2018

- <a name="MGN"></a> **[MGN]** Learning Discriminative Features with Multiple Granularities
for Person Re-Identification (**ACMMM2018**) [[paper](https://dl.acm.org/citation.cfm?id=3240552)]
- <a name="FD-GAN"></a> **[FD-GAN]** FD-GAN: Pose-guided Feature Distilling GAN for Robust Person Re-identification (**NeurIPS2018**) [[paper](http://papers.nips.cc/paper/7398-fd-gan-pose-guided-feature-distilling-gan-for-robust-person-re-identification)]
- <a name="Multi-pseudo"></a> **[Multi-pseudo]** Multi-pseudo regularized label for generated data in person re-identification (**TIP2018**) [[paper](https://ieeexplore.ieee.org/abstract/document/8485730/)] [[arxiv](https://arxiv.org/pdf/1801.06742.pdf)]
- <a name="PAN"></a> **[PAN]** Pedestrian Alignment Network for Large-scale Person Re-identification (**T-CSVT2018**) [[paper](https://ieeexplore.ieee.org/abstract/document/8481710)] [[arxiv](https://arxiv.org/pdf/1707.00408.pdf)]
- <a name=""></a> Person Transfer GAN to Bridge Domain Gap for Person Re-Identification (**CVPR2018**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Person_Transfer_GAN_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1711.08565)]
- <a name=""></a> Disentangled Person Image Generation (**CVPR2018**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Ma_Disentangled_Person_Image_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1712.02621)]
- <a name=""></a> Group Consistent Similarity Learning via Deep CRF for Person Re-Identification (**CVPR2018**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Group_Consistent_Similarity_CVPR_2018_paper.html)]
- <a name=""></a> Diversity Regularized Spatiotemporal Attention for Video-Based Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Diversity_Regularized_Spatiotemporal_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1803.09882)]
- <a name=""></a> A Pose-Sensitive Embedding for Person Re-Identification With Expanded Cross Neighborhood Re-Ranking (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Sarfraz_A_Pose-Sensitive_Embedding_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1711.10378)]
- <a name=""></a> Image-Image Domain Adaptation With Preserved Self-Similarity and Domain-Dissimilarity for Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1711.07027)]
- <a name=""></a> Human Semantic Parsing for Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Kalayeh_Human_Semantic_Parsing_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1804.00216)]
- <a name=""></a> Video Person Re-Identification With Competitive Snippet-Similarity Aggregation and Co-Attentive Snippet Embedding (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Video_Person_Re-Identification_CVPR_2018_paper.html)]
- <a name=""></a> Mask-Guided Contrastive Attention Model for Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Song_Mask-Guided_Contrastive_Attention_CVPR_2018_paper.html)]
- <a name=""></a> Person Re-Identification With Cascaded Pairwise Convolutions (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Person_Re-Identification_With_CVPR_2018_paper.html)]
- <a name="MLFN"></a> **[MLFN]** Multi-Level Factorisation Net for Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Chang_Multi-Level_Factorisation_Net_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1803.09132)]
- <a name=""></a> Attention-Aware Compositional Network for Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Xu_Attention-Aware_Compositional_Network_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1805.03344)]
- <a name=""></a> Deep Group-Shuffling Random Walk for Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Shen_Deep_Group-Shuffling_Random_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1807.11178)]
- <a name="HA-CNN"></a> **[HA-CNN]** Harmonious Attention Network for Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Harmonious_Attention_Network_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1802.08122)]
- <a name=""></a> Efficient and Deep Person Re-Identification Using Multi-Level Similarity (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Guo_Efficient_and_Deep_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1803.11353)]
- <a name="PT"></a> **[PT]** Pose Transferrable Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Pose_Transferrable_Person_CVPR_2018_paper.html)]
- <a name=""></a> Adversarially Occluded Samples for Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Huang_Adversarially_Occluded_Samples_CVPR_2018_paper.html)]
- <a name=""></a> Camera Style Adaptation for Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhong_Camera_Style_Adaptation_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1711.10295)]
- <a name=""></a> Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Exploit_the_Unknown_CVPR_2018_paper.html)]
- <a name=""></a> Dual Attention Matching Network for Context-Aware Feature Sequence Based Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Si_Dual_Attention_Matching_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1803.09937)]
- <a name=""></a> Easy Identification From Better Constraints: Multi-Shot Person Re-Identification From Reference Constraints (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_Easy_Identification_From_CVPR_2018_paper.html)]
- <a name=""></a> Eliminating Background-Bias for Robust Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Tian_Eliminating_Background-Bias_for_CVPR_2018_paper.html)]
- <a name=""></a> End-to-End Deep Kronecker-Product Matching for Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Shen_End-to-End_Deep_Kronecker-Product_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1807.11182)]
- <a name=""></a> Exploiting Transitivity for Learning Person Re-Identification Models on a Budget (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Roy_Exploiting_Transitivity_for_CVPR_2018_paper.html)]
- <a name=""></a> Deep Spatial Feature Reconstruction for Partial Person Re-Identification: Alignment-Free Approach (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/He_Deep_Spatial_Feature_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1801.00881)]
- <a name=""></a> Resource Aware Person Re-Identification Across Multiple Resolutions (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Resource_Aware_Person_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1805.08805)]
- <a name="DeformGAN"></a> **[DeformGAN]** Deformable GANs for Pose-Based Human Image Generation (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Siarohin_Deformable_GANs_for_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1801.00055)]
- <a name=""></a> Maximum Margin Metric Learning Over Discriminative Nullspace for Person Re-identification (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/T_M_Feroz_Ali_Maximum_Margin_Metric_ECCV_2018_paper.html)]
- <a name=""></a> RCAA: Relational Context-Aware Agents for Person Search (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Xiaojun_Chang_RCAA_Relational_Context-Aware_ECCV_2018_paper.html)]
- <a name=""></a> Generalizing A Person Retrieval Model Hetero- and Homogeneously (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Zhun_Zhong_Generalizing_A_Person_ECCV_2018_paper.html)]
- <a name=""></a> Person Search in Videos with One Portrait Through Visual and Temporal Links (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Qingqiu_Huang_Person_Search_in_ECCV_2018_paper.html)] [[github](https://github.com/hqqasw/person-search-PPCC)]
- <a name=""></a> Person Search by Multi-Scale Matching (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Xu_Lan_Person_Search_by_ECCV_2018_paper.html)]
- <a name=""></a> Person Re-identification with Deep Similarity-Guided Graph Neural Network (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Yantao_Shen_Person_Re-identification_with_ECCV_2018_paper.html)]
- <a name="PN-GAN"></a> **[PN-GAN]** Pose-Normalized Image Generation for Person Re-identification (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Xuelin_Qian_Pose-Normalized_Image_Generation_ECCV_2018_paper.html)]
- <a name=""></a> Person Search via A Mask-guided Two-stream CNN Model (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Di_Chen_Person_Search_via_ECCV_2018_paper.html)]
- <a name=""></a> Improving Deep Visual Representation for Person Re-identification by Global and Local Image-language Association (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Dapeng_Chen_Improving_Deep_Visual_ECCV_2018_paper.html)]
- <a name=""></a> Hard-Aware Point-to-Set Deep Metric for Person Re-identification (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Rui_Yu_Hard-Aware_Point-to-Set_Deep_ECCV_2018_paper.html)]
- <a name=""></a> Reinforced Temporal Attention and Split-Rate Transfer for Depth-Based Person Re-Identification (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Nikolaos_Karianakis_Reinforced_Temporal_Attention_ECCV_2018_paper.html)]
- <a name=""></a> Adversarial Open-World Person Re-Identification (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Xiang_Li_Adversarial_Open-World_Person_ECCV_2018_paper.html)]
- <a name="Part-aligned"></a> **[Part-aligned]** Part-Aligned Bilinear Representations for Person Re-Identification (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Yumin_Suh_Part-Aligned_Bilinear_Representations_ECCV_2018_paper.html)]
- <a name="Mancs"></a> **[Mancs]** Mancs: A Multi-task Attentional Network with Curriculum Sampling for Person Re-identification (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Cheng_Wang_Mancs_A_Multi-task_ECCV_2018_paper.html)]
- <a name="PCB"></a> **[PCB]** Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline) (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Yifan_Sun_Beyond_Part_Models_ECCV_2018_paper.html)]

### 2017

- <a name="SHaPE"></a> **[SHaPE]** SHaPE: A Novel Graph Theoretic Algorithm for Making Consensus-Based Decisions in Person Re-Identification Systems (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Barman_SHaPE_A_Novel_ICCV_2017_paper.html)]
- <a name=""></a> Spatio-Temporal Person Retrieval via Natural Language Queries (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Yamaguchi_Spatio-Temporal_Person_Retrieval_ICCV_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1704.07945)]
- <a name=""></a> A Two Stream Siamese Convolutional Neural Network for Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Chung_A_Two_Stream_ICCV_2017_paper.html)]
- <a name=""></a> Efficient Online Local Metric Adaptation via Negative Samples for Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Efficient_Online_Local_ICCV_2017_paper.html)]
- <a name=""></a> Learning View-Invariant Features for Person Identification in Temporally Synchronized Videos Taken by Wearable Cameras (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Zheng_Learning_View-Invariant_Features_ICCV_2017_paper.html)]
- <a name=""></a> Deeply-Learned Part-Aligned Representations for Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Zhao_Deeply-Learned_Part-Aligned_Representations_ICCV_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1707.07256)]
- <a name="LSRO"></a> **[LSRO]** Unlabeled Samples Generated by GAN Improve the Person Re-Identification Baseline in Vitro (**ICCV2017**)(**Spotlight**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Zheng_Unlabeled_Samples_Generated_ICCV_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1701.07717)]
- <a name=""></a> Pose-Driven Deep Convolutional Model for Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Su_Pose-Driven_Deep_Convolutional_ICCV_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1709.08325)]
- <a name=""></a> Jointly Attentive Spatial-Temporal Pooling Networks for Video-Based Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Xu_Jointly_Attentive_Spatial-Temporal_ICCV_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1708.02286)]
- <a name=""></a> RGB-Infrared Cross-Modality Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Wu_RGB-Infrared_Cross-Modality_Person_ICCV_2017_paper.html)]
- <a name=""></a> Multi-Scale Deep Learning Architectures for Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Qian_Multi-Scale_Deep_Learning_ICCV_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1709.05165)]
- <a name="SVDNet"></a> **[SVDNet]** SVDNet for Pedestrian Retrieval (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Sun_SVDNet_for_Pedestrian_ICCV_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1703.05693)]
- <a name="DCF"></a> **[DCF]** Learning Deep Context-Aware Features Over Body and Latent Parts for Person Re-Identification (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Li_Learning_Deep_Context-Aware_CVPR_2017_paper.html)]
- <a name=""></a> Beyond Triplet Loss: A Deep Quadruplet Network for Person Re-Identification (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Chen_Beyond_Triplet_Loss_CVPR_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1704.01719)]
- <a name=""></a> Spindle Net: Person Re-Identification With Human Body Region Guided Feature Decomposition and Fusion (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Spindle_Net_Person_CVPR_2017_paper.html)]
- <a name=""></a> Re-Ranking Person Re-Identification With k-Reciprocal Encoding (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1701.08398)]
- <a name=""></a> Person Re-Identification in the Wild (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Zheng_Person_Re-Identification_in_CVPR_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1604.02531)]
- <a name="SSM"></a> **[SSM]** Scalable Person Re-Identification on Supervised Smoothed Manifold (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Bai_Scalable_Person_Re-Identification_CVPR_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1703.08359)]
- <a name=""></a> One-Shot Metric Learning for Person Re-Identification (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Bak_One-Shot_Metric_Learning_CVPR_2017_paper.html)]
- <a name=""></a> Joint Detection and Identification Feature Learning for Person Search (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Xiao_Joint_Detection_and_CVPR_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1604.01850)]
- <a name=""></a> Multiple People Tracking by Lifted Multicut and Person Re-Identification (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Tang_Multiple_People_Tracking_CVPR_2017_paper.html)]
- <a name=""></a> Point to Set Similarity Based Deep Feature Learning for Person Re-Identification (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Point_to_Set_CVPR_2017_paper.html)]
- <a name=""></a> Fast Person Re-Identification via Cross-Camera Semantic Binary Transformation (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Chen_Fast_Person_Re-Identification_CVPR_2017_paper.html)]
- <a name=""></a> See the Forest for the Trees: Joint Spatial and Temporal Recurrent Neural Networks for Video-Based Person Re-Identification (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_See_the_Forest_CVPR_2017_paper.html)]
- <a name=""></a> Consistent-Aware Deep Learning for Person Re-Identification in a Camera Network (**CVPR2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Consistent-Aware_Deep_Learning_CVPR_2017_paper.html)]


## Leaderboard
The section is being continually updated. Note that some values have superscript, which indicates their source.

### Market-1501

| Year-Conference/Journal | Method | Rank@1 | mAP |
| --- | --- | --- | --- |
| 2017--CVPR | [DCF](#DCF)                | 80.3 | 57.5 |
| 2017--CVPR | [SSM](#SSM)                | 82.2 | 68.8 |
| 2017--ICCV | [SVDNet](#SVDNet)          | 82.3 | 62.1 |
| 2017--ICCV| [LSRO](#LSRO)               | 84.0 | 66.1 |
| 2018--CVPR| [DeformGAN](#DeformGAN)     | 80.6 | 61.3 |
| 2018--T-CSVT| [PAN](#PAN)               | 82.8 | 63.4 |
| 2018--TIP| [Multi-pseudo](#Multi-pseudo)| 85.8 | 67.5 |
| 2018--CVPR| [PT](#PT)                   | 87.7 | 68.9 |
| 2018--ECCV| [PN-GAN](#PN-GAN)           | 89.4 | 72.6 |
| 2018--CVPR| [MLFN](#MLFN)               | 90.0 | 74.3 |
| 2018--NeurIPS| [FD-GAN](#FD-GAN)           | 90.5 | 77.7 |
| 2018--CVPR| [HA-CNN](#HA-CNN)           | 91.2 | 75.7 |
| 2018--ECCV| [Part-aligned](#Part-aligned) | 91.7 | 79.6 |
| 2018--ECCV| [Mancs](#Mancs)             | 93.1 | 82.3 |
| 2018--ACMMM| [MGN](#MGN)             | 95.7 | 86.9 |
| 2019--PR| [AlignedReID++](#AlignedReID++)   | 91.0 | 77.6 |
| 2019--CVPR| [VPM](#VPM)           | 93.0 | 80.8 |
| 2019--CVPR| [AANet](#AANet)           | 93.93 | 83.41 |
| 2019--AAAI| [HPM](#HPM)           | 94.2 | 82.7  |
| 2019--CVPR| [CASN](#CASN)           | 94.4 | 82.8 |
| 2019--CVPR| [IANet](#IANet)         | 94.4 | 83.1 |
| 2019--CVPR| [CAMA](#CAMA)           | 94.7 | 84.5 |
| 2019--CVPR| [DG-Net](#DG-Net)           | 94.8 | 86.0 |
| 2019--ICCV| [ABD-Net](#ABD-Net)           | 95.60 | **88.28** |
| 2019--CVPR| [DSA-reID](#DSA-reID)           | 95.7 | 87.6 |
| 2019--CVPR| [Pyramid](#Pyramid)           | 95.7 | 88.2 |
| 2019--AAAI| [st-ReID](#st-ReID)           | **97.2** | 86.7 |

### DukeMTMC-reID

| Year-Conference/Journal | Method | Rank@1 | mAP |
| --- | --- | --- | --- |
| 2018--NeurIPS| [FD-GAN](#FD-GAN)           | 80.0 | 64.5 |
| 2018--ECCV| [Mancs](#Mancs)             | 84.9 | 71.8 |
| 2018--ACMMM| [MGN](#MGN)             | 88.7 | 78.4 |
| 2019--PR| [AlignedReID++](#AlignedReID++)  | 80.7 | 68.0 |
| 2019--CVPR| [VPM](#VPM)           | 83.6 | 72.6 |
| 2019--CVPR| [CAMA](#CAMA)           | 85.8 | 72.9 |
| 2019--CVPR| [DSA-reID](#DSA-reID)           | 86.2 | 74.3 |
| 2019--AAAI| [HPM](#HPM)           | 86.6 | 74.3 |
| 2019--CVPR| [DG-Net](#DG-Net)           | 86.6 | 74.8 |
| 2019--CVPR| [IANet](#IANet)         | 87.1 |  73.4 |
| 2019--CVPR| [AANet](#AANet)           | 87.65 | 74.29 |
| 2019--CVPR| [CASN](#CASN)           | 87.7 | 73.7 |
| 2019--ICCV| [ABD-Net](#ABD-Net)           | 89.0 | 78.59 |
| 2019--CVPR| [Pyramid](#Pyramid)           | 89.0 | 79.0 |
| 2019--AAAI| [st-ReID](#st-ReID)           | **94.0** | **82.8** |

### MSMT17

| Year-Conference/Journal | Method | Rank@1 | mAP |
| --- | --- | --- | --- |
| 2019--CVPR| [IANet](#IANet)           | 75.5 | 46.8 |
| 2019--CVPR| [DG-Net](#DG-Net)           | **77.2** | **52.3** |

## UDA

### Market-1501

| Year-Conference/Journal | Method | Rank@1 | Rank@5 | Rank@10 | mAP |
| --- | --- | --- | --- | --- | --- |
| 2019--ICCV| [UGA](#UGA)    | 87.2 |      |      | 70.3 |
| 2019--ICCV| [SSG++](#SSG++)| 86.2 | 94.6 | 96.5 | 68.7 |
| 2018--arXiv|[UDAP](#UDAP)  | 75.8 | 89.5 | 93.2 | 53.7 |
| 2019--ICCV| [PDA-Net](#PDA-Net)| 75.2 | 86.3 | 90.2 | 47.6 |
| 2019--CVPR| [ECN](#ECN)    | 75.1 | 87.6 | 91.6 | 43.0 |
| 2019--ICME| [CDs](#CDs)    | 71.6 | 81.2 | 84.7 | 39.9 |
| 2018--CVPR| [ARN](#ARN)    | 70.3 | 80.4 | 86.3 | 39.4 |
| 2019--PAMI| [UTAL](#UTAL)  | 69.2 |      |      | 46.2 |
| 2019--CVPR| [PAUL](#PAUL)  | 68.5 | 82.4 | 87.4 | 40.1 |
| 2019--CVPR| [MAR](#MAR)    | 67.7 | 81.9 |      | 40.0 |
| 2018--ECCV| [DASy](#DAsy)  | 65.7 |      |      |      |
| 2019--ICCV| [CFCL](#CFCL)  | 65.4 | 80.6 | 86.2 | 35.5 |
| 2019--ICCV| [UCDA-CCE](#UCDA-CCE)| 64.3 ||      | 34.5 |
| 2018--ECCV| [TAUDL](#TAUDL)| 63.7 |      |      | 41.2 |
| 2018--ECCV| [HHL](#HHL)    | 62.2 | 78.8 | 84.0 | 31.4 |
| 2019--AAAI| [BUC](#BUC)    | 61.9 | 73.5 | 78.2 | 29.6 |
| 2019--PAMI| [DECAMEL](#DECAMEL)| 60.2 | 76.0 |  | 32.4 |
| 2019--TIP | [CamStyle](#CamStyle)| 58.8 | 78.2 | 84.3 | 31.4 |
| 2018--CVPR| [TJ-AIDL](#TJ-AIDL)| 58.2 | 74.8 | 81.1 | 26.5 |
| 2018--CVPR| [SPGAN+LMP](#SPGAN+LMP)| 57.7 | 75.8 | 82.4 | 26.7 |
| 2018--BMVC| [MMFA](#MMFA)  | 56.7 | 75.0 | 81.8 | 27.4 |
| 2017--ICCV| [CAMEL](#CAMEL)| 54.5 | 73.1 |      | 26.3 |
| 2018--CVPR| [SPGAN](#SPGAN)| 51.5 | 70.1 | 76.8 | 27.1 |

### DukeMTMC-reID

| Year-Conference/Journal | Method | Rank@1 | Rank@5 | Rank@10 | mAP |
| --- | --- | --- | --- | --- | --- |
| 2019--ICCV| [SSG++](#SSG++)| 76.0 | 85.8 | 89.3 | 60.3 |
| 2019--ICCV| [UGA](#UGA)    | 75.0 |      |      | 53.3 |
| 2019--CVPR| [PAUL](#PAUL)  | 72.0 | 82.7 | 86.0 | 53.2 |
| 2019--PAMI| [UTAL](#UTAL)  | 69.2 |      |      | 46.2 |
| 2018--arXiv|[UDAP](#UDAP)  | 68.4 | 80.1 | 83.5 | 49.0 |
| 2019--ICME| [CDs](#CDs)    | 67.2 | 75.9 | 79.4 | 42.7 |
| 2019--CVPR| [MAR](#MAR)    | 67.1 | 79.8 |      | 48.0 |
| 2019--CVPR| [ECN](#ECN)    | 63.3 | 75.8 | 80.4 | 40.4 |
| 2019--ICCV| [PDA-Net](#PDA-Net)| 63.2 | 77.0 | 82.5 | 45.1 |
| 2018--ECCV| [TAUDL](#TAUDL)| 61.7 |      |      | 43.5 |
| 2018--CVPR| [ARN](#ARN)    | 60.2 | 73.9 | 79.5 | 33.4 |
| 2019--ICCV| [CFCL](#CFCL)  | 59.3 | 73.2 | 77.8 | 37.8 |
| 2019--ICCV| [UCDA-CCE](#UCDA-CCE)| 55.4 ||      | 36.7 |
| 2019--TIP | [CamStyle](#CamStyle)| 48.4 | 62.5 | 68.9 | 25.1|
| 2018--ECCV| [HHL](#HHL)    | 46.9 | 61.0 | 66.7 | 27.2 |
| 2018--CVPR| [SPGAN+LMP](#SPGAN+LMP)| 46.4 | 62.3 | 68.0 | 26.2 |
| 2018--BMVC| [MMFA](#MMFA)  | 45.3 | 59.8 | 66.3 | 24.7 |
| 2018--CVPR| [TJ-AIDL](#TJ-AIDL)| 44.3 | 59.6 | 65.0 | 23.0 |
| 2018--CVPR| [SPGAN](#SPGAN)| 41.1 | 56.6 | 63.0 | 22.3 |
| 2019--AAAI| [BUC](#BUC)    | 40.4 | 52.5 | 58.2 | 22.1 |
| 2017--ICCV| [CAMEL](#CAMEL)| 40.3 | 57.6 |      | 19.8 |
| 2017--ICCV| [CycleGAN](#CycleGAN)| 38.5 ||      | 19.9 |
| 2018--TOMM| [PUL](#PUL)    | 30.0 | 43.4 | 48.5 | 16.4 |
| 2018--CVPR| [PTGAN](#PTGAN)| 27.4 | 43.6 | 50.7 | 13.5 |

### PRID2011

| Year-Conference/Journal | Method | Rank@1 | Rank@5 | Rank@10 | Rank@20 |
| --- | --- | --- | --- | --- | --- |
| 2019--Access| [proposed](#proposed)| 91.7 | 96.7 |      | 98.7 |
| 2018--BMVC| [DAL](#DAL)   | 85.3 | 97.0 |      | 99.6 |
| 2019--TIP | [DGM+](#DGM+) | 81.4 | 95.8 | 98.3 | 99.6 |
| 2017--ICCV| [SMP*](#SMP*) | 80.9 | 93.3 | 97.8 | 99.4 |
| 2017--ICCV| [DGM+MLAPG](#DGM+MLAPG)| 73.1 | 92.5 |      | 99.0 |
| 2017--WACV| [PAM+LOMO](#PAM+LOMO)| 70.6 | 90.2 ||97.1 |
| 2017--ICCV| [DGM+IDE](#DGM+IDE)| 56.4 | 81.3 | | 96.4 |
| 2019--PAMI| [UTAL](#UTAL) | 54.7 | 83.1 |      | 96.2 |
| 2018--ECCV| [RACE](#RACE) | 50.6 | 79.4 |      | 91.8 |
| 2018--ECCV| [TAUDL](#TAUDL)|49.4 | 78.7 |      | 98.9 |
| 2018--ECCV| [DAsy](#DAsy) | 43.0 ||||
| 2017--PR  | [MDTS](#MDTS) | 41.7 | 67.1 | 79.4 | 90.1 |

### iLIDS-VID
| Year-Conference/Journal | Method | Rank@1 | Rank@5 | Rank@10 | Rank@20 |
| --- | --- | --- | --- | --- | --- |
| 2019--Access| [proposed](#proposed)| 79.1 | 93.5 || 97.5 |
| 2018--BMVC| [DAL](#DAL)   | 56.9 | 80.6 || 91.9 |
| 2018--ECCV| [DAsy](#DAsy) | 56.5 ||||
| 2017--ICCV| [SMP*](#SMP*) | 41.7 | 66.3 | 74.1 | 80.7 |
| 2017--ICCV| [DGM+MLAPG](#DGM+MLAPG)| 37.1 | 61.3 || 82.0 |
| 2019--PAMI| [UTAL](#UTAL) | 35.1 | 59.0 |      | 83.8 |
| 2017--WACV| [PAM+LOMO](#PAM+LOMO)| 33.3 | 57.8 ||80.5 |
| 2017--PR  | [MDTS](#MDTS) | 31.5 | 62.1 | 72.8 | 82.4 |
| 2018--ECCV| [TAUDL](#TAUDL)|26.7 | 51.3 |      | 82.0 |
| 2018--ECCV| [RACE](#RACE) | 19.3 | 39.3 |      | 68.7 |

### MARS
| Year-Conference/Journal | Method | Rank@1 | Rank@5 | Rank@20 | mAP |
| --- | --- | --- | --- | --- | --- |
| 2019--PAMI| [UTAL](#UTAL) | 49.9 | 66.4 | 77.8 | 35.2 |
| 2018--BMVC| [DAL](#DAL)   | 46.8 | 63.9 | 77.5 | 21.4 |
| 2018--ECCV| [TAUDL](#TAUDL)|43.8 | 59.9 | 72.8 | 29.1 |
| 2018--ECCV| [RACE](#RACE) | 43.2 | 57.1 | 67.6 | 24.5 |
| 2019--Access| [proposed](#proposed)| 39.7 | 53.2 | 64.1 | 20.1 |
| 2017--ICCV| [DGM+MLAPG](#DGM+MLAPG)| 24.6 | 42.6 | 57.2 | 11.8 |
| 2017--ICCV| [SMP*](#SMP*) | 23.9 | 35.8 | 44.9 | 10.5 |
