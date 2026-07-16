这个项目在做的是基于 SurroundOcc 复现基础，构建一套可迁移到自采数据的 Occ3D/FlashOcc 风格 occupancy GT 生成流水线，重点解决 dense occupancy supervision 中的 visibility reasoning、free/unknown 区分、camera-visible mask 和语义融合问题。尝试把 Occ3D 半开源论文里的思想工程化、可验证化，并且朝自采数据迁移。

相关数据解释见 docs/data_inventory.md

实验记录见 docs/Experiment_records.md

运行命令见 docs/Run_Commands_for_Process.md