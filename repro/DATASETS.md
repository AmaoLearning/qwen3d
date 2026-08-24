# Dataset inventory

Qwen-3D does not have a single downloadable evaluation dataset. The upstream
recipe combines assets governed by different licenses.

| Asset | Local path | Acquisition |
|---|---|---|
| ScanEnts3D Nr3D/ScanRefer annotations | `data/refer_it_3d` | Public, downloaded by the helper |
| Sr3D annotations | `data/refer_it_3d` | Public Google Drive folder, downloaded by the helper |
| UniVLG precomputed ScanNet metadata | `data/scannet_precomputed` | Public Hugging Face snapshot |
| ScanNet RGB-D/scans | `data/posed_rgbd` | Requires accepting ScanNet Terms of Use and requesting official download access |
| ScanNet200 processed database | `data/mask3d_processed/scannet200` | Must be generated from licensed ScanNet scans using upstream/Mask3D preparation |
| ScanQA and SQA3D annotations | `data/refer_it_3d` | Public Google Drive/Zenodo files, normalized by `download_qa_data.py` |
| Matterport3D | `data/posed_rgbd` | Requires a separate Matterport3D license agreement |
| COCO 2017 + RefCOCO/+/g + LLaVA-150K | `data/datasets_2d` | Needed only for joint 2D-3D training, not 3D grounding evaluation |

The automated helper deliberately does not bypass dataset agreements. Once the
licensed ScanNet archive is placed in `data/posed_rgbd`, follow upstream
`docs/DATA.md` and the linked UniVLG/Mask3D preprocessing to produce the YAML
specified by `SCANNET_DATA_DIR`.
