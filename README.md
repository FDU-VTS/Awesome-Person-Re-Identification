# Awesome Person Re-Identification

If you have any problems, suggestions or improvements, please submit the issue or PR.

## Contents
* [Datasets](#datasets)
* [Papers](#papers)
* [Leaderboard](#leaderboard)


<!-- ### Code
- [[C^3 Framework](https://github.com/gjy3035/C-3-Framework)] An open-source PyTorch code for crowd counting, which is released. -->

<!-- ### Technical blog
- [2019.05] [Chinese Blog] C^3 Framework系列之一：一个基于PyTorch的开源人群计数框架 [[Link](https://zhuanlan.zhihu.com/p/65650998)]
- [2019.04] Crowd counting from scratch [[Link](https://github.com/CommissarMa/Crowd_counting_from_scratch)]
- [2017.11] Counting Crowds and Lines with AI [[Link1](https://blog.dimroc.com/2017/11/19/counting-crowds-and-lines/)] [[Link2](https://count.dimroc.com/)] [[Code](https://github.com/dimroc/count)] -->

<!-- ###  GT generation
- Density Map Generation from Key Points [[Matlab Code](https://github.com/aachenhang/crowdcount-mcnn/tree/master/data_preparation)] [[Python Code](https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/make_dataset.ipynb)] [[Fast Python Code](https://github.com/vlad3996/computing-density-maps)] -->


## Datasets

| Dataset                   | Release time     | # identities | # cameras   | # images |
|---------------------------|------------------|--------------|-------------|----------|
| [VIPeR](https://vision.soe.ucsc.edu/node/178)                     | 2007             | 632          | 2           | 1264     |
| [ETH1,2,3](http://homepages.dcc.ufmg.br/~william/datasets.html)                  | 2007             | 85, 35, 28     | 1           | 8580     |
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
| [CASIA Gait Database B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)     | 2015(*see below) | 124          | 11          |          |
| [Market1501](http://www.liangzheng.org/Project/project_reid.html)                | 2015             | 1501         | 6           | 32217    |
| [PKU-Reid](https://github.com/charliememory/PKU-Reid-Dataset)                  | 2016             | 114          | 2           | 1824     |
| [PRW](http://www.liangzheng.org/Project/project_prw.html)                       | 2016             | 932          | 6           | 34304    |
| [Large scale person search](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html) | 2016             | 11934s       | -           | 34574    |                    |
| [MARS](http://www.liangzheng.com.cn/Project/project_mars.html)                      | 2016             | 1261         | 6           | 1191003  |
| [DukeMTMC-reID](http://vision.cs.duke.edu/DukeMTMC/)             | 2017             | 1812         | 8           | 36441    |                       |
| [DukeMTMC4ReID](http://vision.cs.duke.edu/DukeMTMC/)             | 2017             | 1852         | 8           | 46261    |                      |
| [Airport](http://www.northeastern.edu/alert/transitioning-technology/alert-datasets/alert-airport-re-identification-dataset/)                   | 2017             | 9651         | 6           | 39902    |
| [MSMT17](http://www.pkuvmc.com/publications/msmt17.html)                    | 2018             | 4101         | 15          | 126441   |
| [RPIfield](https://drive.google.com/file/d/1GO1zm7vCAJwXgJtoFyUs367_Knz8Ev0A/view?usp=sharing)   | 2018      | 112       | 12        | 601,581       |

## Papers

### arXiv papers
This section only includes the last ten papers since 2018 in [arXiv.org](arXiv.org). Previous papers will be hidden using  ```<!--...-->```. If you want to view them, please open the [raw file](https://raw.githubusercontent.com/gjy3035/Awesome-Crowd-Counting/master/README.md) to read the source code. Note that all unpublished arXiv papers are not included into [the leaderboard of performance](#performance).

- <a name=""></a> Orthogonal Center Learning with Subspace Masking for Person Re-Identification [[paper](https://arxiv.org/pdf/1908.10535.pdf)]
- <a name=""></a> CDPM: Convolutional Deformable Part Models for Person Re-identification [[paper](https://arxiv.org/abs/1906.04976)]
- <a name=""></a> Rethinking Person Re-Identification with Confidence [[paper](https://arxiv.org/abs/1906.04692)]
- <a name=""></a> Omni-Scale Feature Learning for Person Re-Identification [[paper](https://arxiv.org/abs/1905.00953)]
- <a name=""></a> Group Re-Identification with Multi-grained Matching and Integration [[paper](https://arxiv.org/abs/1905.07108)]
- <a name=""></a> HPILN: A feature learning framework for cross-modality person re-identification [[paper](https://arxiv.org/abs/1906.03142)]
- <a name=""></a> PAC-GAN: An Effective Pose Augmentation Scheme for Unsupervised Cross-View Person Re-identification [[paper](https://arxiv.org/pdf/1906.01792.pdf)]
- <a name=""></a> Towards better Validity: Dispersion based Clustering for Unsupervised Person Re-identification [[paper](https://arxiv.org/abs/1906.01308)] [[github](https://github.com/gddingcs/Dispersion-based-Clustering)]



### Survey
- <a name=""></a> A Survey of Open-World Person Re-identification (**T-CSVT2019**) [[paper](https://ieeexplore.ieee.org/abstract/document/8640834)]
- <a name=""></a> A systematic evaluation and benchmark for person re-identification: Features, metrics, and datasets (**T-PAMI2016**) [[arxiv](https://arxiv.org/abs/1605.09653)]
- <a name=""></a> Person Re-identification: Past, Present and Future (**arXiv2016**) [[arxiv](https://arxiv.org/abs/1610.02984)]
- <a name=""></a> A survey of approaches and trends in person re-identification (**Image and Vision Computing 2014**) [[paper](https://www.sciencedirect.com/science/article/pii/S0262885614000262)]
- <a name=""></a> Appearance Descriptors for Person Re-identification: a Comprehensive Review (**arXiv2013**) [[arxiv](https://arxiv.org/abs/1307.574)]
- <a name=""></a> People reidentification in surveillance and forensics: A survey (**CSUR2013**) [[paper](https://dl.acm.org/citation.cfm?id=2543596)]
- <a name=""></a> Intelligent multi-camera video surveillance: A review (**PR Letters2013**) [[paper](https://www.sciencedirect.com/science/article/pii/S016786551200219X)]

### Methods dealing with the lack of labelled data
- <a name="UTAL"></a> **[UTAL]** Unsupervised Tracklet Person Re-Identification (**T-PAMI2019**) [[paper](https://ieeexplore.ieee.org/abstract/document/8658110)]
- <a name="MAR"></a> **[MAR]** Unsupervised Person Re-identification by Soft Multilabel Learning (**CVPR2019**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Unsupervised_Person_Re-Identification_by_Soft_Multilabel_Learning_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1903.06325.pdf)] [[github](https://github.com/KovenYu/MAR)]
- <a name="E2E"></a> **[E2E]** Unsupervised Person Image Generation with Semantic Parsing Transformation (**CVPR2019**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Song_Unsupervised_Person_Image_Generation_With_Semantic_Parsing_Transformation_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1904.03379.pdf)] [[github](https://github.com/SijieSong/person_generation_spt)]
- <a name=""></a> Patch-Based Discriminative Feature Learning for Unsupervised Person Re-Identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Patch-Based_Discriminative_Feature_Learning_for_Unsupervised_Person_Re-Identification_CVPR_2019_paper.html)]
- <a name=""></a> A Bottom-Up Clustering Approach to Unsupervised Person Re-identification (**AAAI2019**) (**Oral**) [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/4898)] [[github](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification)]
- <a name=""></a> A Novel Unsupervised Camera-aware Domain Adaptation Framework for Person Re-identification (**ICCV2019**) (**Oral**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Qi_A_Novel_Unsupervised_Camera-Aware_Domain_Adaptation_Framework_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1904.03425)]
- <a name=""></a> Weakly Supervised Person Re-Identification (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Meng_Weakly_Supervised_Person_Re-Identification_CVPR_2019_paper.html)] [[arxiv](https://arxiv.org/pdf/1904.03832.pdf)]
- <a name=""></a> Distilled Person Re-identification: Towards a More Scalable System (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Distilled_Person_Re-Identification_Towards_a_More_Scalable_System_CVPR_2019_paper.html)]
- <a name=""></a> Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification (**ICCV2019**) (**Oral**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Fu_Self-Similarity_Grouping_A_Simple_Unsupervised_Cross_Domain_Adaptation_Approach_for_ICCV_2019_paper.html)][[arxiv](https://arxiv.org/abs/1811.10144)]
- <a name=""></a> Transferable Joint Attribute-Identity Deep Learning for Unsupervised Person Re-Identification (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Transferable_Joint_Attribute-Identity_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1803.09786)]
- <a name=""></a> Unsupervised Cross-Dataset Person Re-Identification by Transfer Learning of Spatial-Temporal Patterns (**CVPR2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Lv_Unsupervised_Cross-Dataset_Person_CVPR_2018_paper.html)] [[arxiv](https://arxiv.org/abs/1803.07293)]
- <a name=""></a> Domain Adaptation through Synthesis for Unsupervised Person Re-identification (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper.html)]
- <a name=""></a> Robust Anchor Embedding for Unsupervised Video Person Re-Identification in the Wild (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Mang_YE_Robust_Anchor_Embedding_ECCV_2018_paper.html)]
- <a name=""></a> Unsupervised Person Re-identification by Deep Learning Tracklet Association (**ECCV2018**)(Oral) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Minxian_Li_Unsupervised_Person_Re-identification_ECCV_2018_paper.html)]
- <a name=""></a> Cross-View Asymmetric Metric Learning for Unsupervised Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Yu_Cross-View_Asymmetric_Metric_ICCV_2017_paper.html)] [[arxiv](https://arxiv.org/abs/1708.08062)]
- <a name=""></a> Stepwise Metric Promotion for Unsupervised Video Person Re-Identification (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Stepwise_Metric_Promotion_ICCV_2017_paper.html)]
- <a name=""></a> A Novel Unsupervised Camera-Aware Domain Adaptation Framework for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Qi_A_Novel_Unsupervised_Camera-Aware_Domain_Adaptation_Framework_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1904.03425)]
- <a name=""></a> Self-Training With Progressive Augmentation for Unsupervised Cross-Domain Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Self-Training_With_Progressive_Augmentation_for_Unsupervised_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1907.13315)]
- <a name=""></a> Unsupervised Graph Association for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Unsupervised_Graph_Association_for_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> Unsupervised Person Re-Identification by Camera-Aware Similarity Consistency Learning (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Unsupervised_Person_Re-Identification_by_Camera-Aware_Similarity_Consistency_Learning_ICCV_2019_paper.html)]
- <a name=""></a> Self-Similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-Identification (**ICCV2019**) (**Oral**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Fu_Self-Similarity_Grouping_A_Simple_Unsupervised_Cross_Domain_Adaptation_Approach_for_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1811.10144)]
- <a name=""></a> Cross-Dataset Person Re-Identification via Unsupervised Pose Disentanglement and Adaptation (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Li_Cross-Dataset_Person_Re-Identification_via_Unsupervised_Pose_Disentanglement_and_Adaptation_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1909.09675)]
- <a name=""></a> Deep Reinforcement Active Learning for Human-in-the-Loop Person Re-Identification (**ICCV2019**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Deep_Reinforcement_Active_Learning_for_Human-in-the-Loop_Person_Re-Identification_ICCV_2019_paper.html)]



### 2019

- <a name=""></a>  Learning Disentangled Representation for Robust Person Re-identification (**NeurIPS2019**)
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
- <a name=""></a> Pose-Guided Feature Alignment for Occluded Person Re-Identification (**ICCV2019**) [[paper]http://openaccess.thecvf.com/content_ICCV_2019/html/Miao_Pose-Guided_Feature_Alignment_for_Occluded_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> Robust Person Re-Identification by Modelling Feature Uncertainty (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Yu_Robust_Person_Re-Identification_by_Modelling_Feature_Uncertainty_ICCV_2019_paper.html)]
- <a name=""></a> Co-Segmentation Inspired Attention Networks for Video-Based Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Subramaniam_Co-Segmentation_Inspired_Attention_Networks_for_Video-Based_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> RGB-Infrared Cross-Modality Person Re-Identification via Joint Pixel and Feature Alignment (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Wang_RGB-Infrared_Cross-Modality_Person_Re-Identification_via_Joint_Pixel_and_Feature_Alignment_ICCV_2019_paper.html)]
- <a name=""></a> Beyond Human Parts: Dual Part-Aligned Representations for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Guo_Beyond_Human_Parts_Dual_Part-Aligned_Representations_for_Person_Re-Identification_ICCV_2019_paper.html)]
- <a name=""></a> Batch DropBlock Network for Person Re-Identification and Beyond (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Dai_Batch_DropBlock_Network_for_Person_Re-Identification_and_Beyond_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1811.07130)]
- <a name=""></a> Omni-Scale Feature Learning for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1905.00953)]
- <a name=""></a> Auto-ReID: Searching for a Part-Aware ConvNet for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Quan_Auto-ReID_Searching_for_a_Part-Aware_ConvNet_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1903.09776)]
- <a name=""></a> Second-Order Non-Local Attention Networks for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Xia_Second-Order_Non-Local_Attention_Networks_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1909.00295)]
- <a name=""></a> Global-Local Temporal Representations for Video Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Li_Global-Local_Temporal_Representations_for_Video_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1908.10049)]
- <a name=""></a> Spectral Feature Transformation for Person Re-Identification (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Luo_Spectral_Feature_Transformation_for_Person_Re-Identification_ICCV_2019_paper.html)] [[arxiv](https://arxiv.org/abs/1811.11405)]
- <a name=""></a> A Dual-Path Model With Adaptive Attention for Vehicle Re-Identification (**ICCV2019**)(**Oral**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Khorramshahi_A_Dual-Path_Model_With_Adaptive_Attention_for_Vehicle_Re-Identification_ICCV_2019_paper.html] [[arxiv](https://arxiv.org/abs/1905.03397)]
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
- <a name=""></a> Person Search in Videos with One Portrait Through Visual and Temporal Links (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Qingqiu_Huang_Person_Search_in_ECCV_2018_paper.html)]
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
