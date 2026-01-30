"""
数据加载模块 (Data Loader Module)
加载队友预处理后的engineered_data.csv
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re


class DWTSDataLoader:
    """加载预处理后的DWTS数据"""
    
    def __init__(self, data_path: str):
        """
        初始化数据加载器
        
        Args:
            data_path: 预处理后的CSV文件路径 (engineered_data.csv)
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = {}
        
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        self.raw_data = pd.read_csv(self.data_path)
        print(f"数据加载完成: {self.raw_data.shape[0]} 行 × {self.raw_data.shape[1]} 列")
        return self.raw_data
    
    def get_voting_method(self, season: int) -> str:
        """
        获取该赛季使用的投票结合方法
        
        Args:
            season: 赛季编号
            
        Returns:
            'rank' (排名法) 或 'percentage' (百分比法) 或 'rank_bottom2' (排名法+Bottom Two)
        """
        if season in [1, 2]:
            return 'rank'
        elif season <= 27:
            return 'percentage'
        else:  # season >= 28
            return 'rank_bottom2'
    
    def get_season_max_week(self, season: int) -> int:
        """获取该赛季的最大周数"""
        season_df = self.raw_data[self.raw_data['season'] == season]
        
        # 找到有有效分数的最大周
        max_week = 1
        for week in range(1, 12):
            col = f'week{week}_total_score'
            if col in season_df.columns:
                if season_df[col].notna().any() and (season_df[col] > 0).any():
                    max_week = week
        return max_week
    
    def process_season(self, season: int) -> Dict:
        """
        处理单个赛季的数据，提取每周的评委得分份额和淘汰信息
        
        Args:
            season: 赛季编号
            
        Returns:
            赛季数据字典
        """
        season_df = self.raw_data[self.raw_data['season'] == season].copy()
        season_df = season_df.reset_index(drop=True)
        
        max_week = self.get_season_max_week(season)
        
        season_data = {
            'season': season,
            'voting_method': self.get_voting_method(season),
            'celebrities': season_df['celebrity_name'].tolist(),
            'partners': season_df['ballroom_partner'].tolist(),
            'placements': season_df['placement'].tolist(),
            'max_week': max_week,
            'weeks': {}
        }
        
        # 逐周处理
        for week in range(1, max_week + 1):
            week_data = self._process_week(season_df, week)
            if week_data is not None:
                season_data['weeks'][week] = week_data
        
        return season_data
    
    def _process_week(self, season_df: pd.DataFrame, week: int) -> Dict:
        """处理单周数据"""
        percent_col = f'week{week}_percent_score'
        rank_col = f'week{week}_judge_rank'
        total_col = f'week{week}_total_score'
        
        # 检查列是否存在
        if percent_col not in season_df.columns:
            return None
        
        # 找出该周存活的选手（得分>0）
        scores = season_df[total_col].fillna(0).values
        survivors = np.where(scores > 0)[0].tolist()
        
        if len(survivors) < 2:
            return None
        
        # 获取该周被淘汰的选手
        # 淘汰者是下周分数变为0的人
        eliminated_idx = self._find_eliminated(season_df, week, survivors)
        
        # 计算评委得分份额（使用percent_score进行归一化）
        percent_scores = season_df[percent_col].fillna(0).values
        survivor_scores = percent_scores[survivors]
        
        # 归一化为份额
        total = np.sum(survivor_scores)
        if total == 0:
            judge_share = np.ones(len(survivors)) / len(survivors)
        else:
            judge_share = survivor_scores / total
        
        # 获取评委排名（直接使用预处理的rank）
        if rank_col in season_df.columns:
            judge_ranks = season_df[rank_col].fillna(99).values
            survivor_ranks = judge_ranks[survivors]
        else:
            survivor_ranks = None
        
        return {
            'week_num': week,
            'survivors': survivors,
            'n_survivors': len(survivors),
            'eliminated_idx': eliminated_idx,
            'eliminated_idx_in_survivors': survivors.index(eliminated_idx) if eliminated_idx in survivors else None,
            'judge_share': judge_share,
            'judge_ranks': survivor_ranks,
            'survivor_names': [season_df.iloc[i]['celebrity_name'] for i in survivors]
        }
    
    def _find_eliminated(self, season_df: pd.DataFrame, week: int, survivors: List[int]) -> int:
        """找出该周被淘汰的选手"""
        next_week = week + 1
        next_col = f'week{next_week}_total_score'
        
        if next_col not in season_df.columns:
            # 最后一周，看placement找冠军以外被淘汰的
            return None
        
        next_scores = season_df[next_col].fillna(0).values
        
        # 该周存活但下周分数为0的人就是被淘汰的
        for idx in survivors:
            if next_scores[idx] == 0:
                return idx
        
        return None
    
    def process_all_seasons(self, seasons: List[int] = None) -> Dict:
        """处理所有赛季"""
        if self.raw_data is None:
            self.load_data()
        
        if seasons is None:
            seasons = sorted(self.raw_data['season'].unique())
        
        for season in seasons:
            print(f"处理赛季 {season}...")
            self.processed_data[season] = self.process_season(season)
        
        return self.processed_data
    
    def get_controversy_celebrities(self) -> pd.DataFrame:
        """获取争议选手（有大量fan saves的选手）"""
        if self.raw_data is None:
            self.load_data()
        
        controversy = self.raw_data[
            (self.raw_data['total_fan_saves_bottom1'] > 0) |
            (self.raw_data['total_fan_saves_bottom2'] >= 3)
        ][['celebrity_name', 'season', 'placement', 
           'total_fan_saves_bottom1', 'total_fan_saves_bottom2', 'total_fan_saves_bottom3',
           'avg_judge_rank', 'partner_avg_placement']]
        
        return controversy.sort_values('total_fan_saves_bottom1', ascending=False)


if __name__ == "__main__":
    # 测试代码
    loader = DWTSDataLoader("engineered_data.csv")
    
    try:
        loader.load_data()
        print("\n列名预览:")
        print(loader.raw_data.columns.tolist()[-10:])
        
        # 处理前3个赛季
        data = loader.process_all_seasons(seasons=[1, 2, 3])
        
        print(f"\n成功处理 {len(data)} 个赛季")
        
        # 显示争议选手
        print("\n争议选手（大量粉丝拯救）:")
        controversy = loader.get_controversy_celebrities()
        print(controversy.head(10))
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
