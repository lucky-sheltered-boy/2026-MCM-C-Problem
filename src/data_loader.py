"""
数据加载模块 (Data Loader Module)
加载队友预处理后的engineered_data.csv

处理特殊情况：
1. 某一周多人被淘汰
2. 某一周0人被淘汰
3. 最后一周决出多人顺序（决赛）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
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
    
    def _parse_result(self, result_str: str) -> Tuple[str, Optional[int], Optional[int]]:
        """
        解析results字段
        
        Returns:
            (类型, 周次, 名次)
            类型: 'eliminated', 'finalist', 'withdrew'
        """
        result_str = str(result_str)
        
        # 淘汰
        if 'Eliminated' in result_str or 'eliminated' in result_str:
            match = re.search(r'Week\s*(\d+)', result_str, re.IGNORECASE)
            if match:
                return ('eliminated', int(match.group(1)), None)
        
        # 退赛
        if 'Withdrew' in result_str or 'withdrew' in result_str:
            return ('withdrew', None, None)
        
        # 决赛名次
        place_match = re.search(r'(\d+)(st|nd|rd|th)\s*Place', result_str, re.IGNORECASE)
        if place_match:
            return ('finalist', None, int(place_match.group(1)))
        
        return ('unknown', None, None)
    
    def get_season_max_week(self, season: int) -> int:
        """获取该赛季的最大周数（有有效分数的最大周）"""
        season_df = self.raw_data[self.raw_data['season'] == season]
        
        max_week = 1
        for week in range(1, 12):
            col = f'week{week}_total_score'
            if col in season_df.columns:
                if season_df[col].notna().any() and (season_df[col] > 0).any():
                    max_week = week
        return max_week
    
    def _get_elimination_map(self, season_df: pd.DataFrame) -> Dict[int, List[int]]:
        """
        构建淘汰映射：week -> [被淘汰选手索引列表]
        
        处理：
        - 多人淘汰
        - 退赛
        """
        elim_map = {}  # week -> [indices]
        
        for idx, row in season_df.iterrows():
            result_type, week, place = self._parse_result(row['results'])
            
            if result_type == 'eliminated' and week is not None:
                if week not in elim_map:
                    elim_map[week] = []
                elim_map[week].append(idx)
        
        return elim_map
    
    def _get_finalists(self, season_df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        获取决赛选手及其名次
        
        Returns:
            [(选手索引, 名次), ...] 按名次排序
        """
        finalists = []
        
        for idx, row in season_df.iterrows():
            result_type, week, place = self._parse_result(row['results'])
            
            if result_type == 'finalist' and place is not None:
                finalists.append((idx, place))
        
        return sorted(finalists, key=lambda x: x[1])
    
    def process_season(self, season: int) -> Dict:
        """
        处理单个赛季的数据
        
        Args:
            season: 赛季编号
            
        Returns:
            赛季数据字典
        """
        season_df = self.raw_data[self.raw_data['season'] == season].copy()
        season_df = season_df.reset_index(drop=True)
        
        max_week = self.get_season_max_week(season)
        elim_map = self._get_elimination_map(season_df)
        finalists = self._get_finalists(season_df)
        
        season_data = {
            'season': season,
            'voting_method': self.get_voting_method(season),
            'celebrities': season_df['celebrity_name'].tolist(),
            'partners': season_df['ballroom_partner'].tolist(),
            'placements': season_df['placement'].tolist(),
            'results': season_df['results'].tolist(),
            'max_week': max_week,
            'finalists': finalists,  # [(idx, place), ...]
            'elimination_map': elim_map,  # week -> [indices]
            'weeks': {}
        }
        
        # 逐周处理
        for week in range(1, max_week + 1):
            week_data = self._process_week(season_df, week, elim_map, finalists, max_week)
            if week_data is not None:
                season_data['weeks'][week] = week_data
        
        return season_data
    
    def _process_week(self, season_df: pd.DataFrame, week: int, 
                      elim_map: Dict[int, List[int]], 
                      finalists: List[Tuple[int, int]],
                      max_week: int) -> Dict:
        """
        处理单周数据
        
        处理特殊情况：
        - 多人淘汰：eliminated_indices为列表
        - 0人淘汰：eliminated_indices为空列表，is_no_elimination=True
        - 最后一周（决赛）：is_finale=True，包含决赛排名信息
        """
        percent_col = f'week{week}_percent_score'
        rank_col = f'week{week}_judge_rank'
        total_col = f'week{week}_total_score'
        
        # 检查列是否存在
        if total_col not in season_df.columns:
            return None
        
        # 找出该周存活的选手（得分>0）
        scores = season_df[total_col].fillna(0).values
        survivors = np.where(scores > 0)[0].tolist()
        
        if len(survivors) < 2:
            return None
        
        # 获取该周被淘汰的选手列表
        eliminated_indices = elim_map.get(week, [])
        
        # 判断是否是决赛周（最后一周）
        is_finale = (week == max_week)
        
        # 计算评委得分份额
        if percent_col in season_df.columns:
            percent_scores = season_df[percent_col].fillna(0).values
            survivor_scores = percent_scores[survivors]
        else:
            survivor_scores = scores[survivors]
        
        # 归一化为份额
        total = np.sum(survivor_scores)
        if total == 0:
            judge_share = np.ones(len(survivors)) / len(survivors)
        else:
            judge_share = survivor_scores / total
        
        # 获取评委排名
        if rank_col in season_df.columns:
            judge_ranks = season_df[rank_col].fillna(99).values
            survivor_ranks = judge_ranks[survivors]
        else:
            survivor_ranks = None
        
        # 计算淘汰选手在存活者中的索引
        eliminated_indices_in_survivors = []
        for elim_idx in eliminated_indices:
            if elim_idx in survivors:
                eliminated_indices_in_survivors.append(survivors.index(elim_idx))
        
        week_data = {
            'week_num': week,
            'survivors': survivors,
            'n_survivors': len(survivors),
            'survivor_names': [season_df.iloc[i]['celebrity_name'] for i in survivors],
            
            # 淘汰信息（支持多人淘汰）
            'eliminated_indices': eliminated_indices,  # 原始索引
            'eliminated_indices_in_survivors': eliminated_indices_in_survivors,  # 在存活者中的索引
            'n_eliminated': len(eliminated_indices),
            'is_no_elimination': len(eliminated_indices) == 0,
            'is_multi_elimination': len(eliminated_indices) > 1,
            
            # 评委信息
            'judge_share': judge_share,
            'judge_ranks': survivor_ranks,
            
            # 决赛信息
            'is_finale': is_finale,
        }
        
        # 如果是决赛周，添加决赛排名信息
        if is_finale:
            # 筛选出在本周存活的决赛选手
            finale_rankings = []
            for idx, place in finalists:
                if idx in survivors:
                    survivor_idx = survivors.index(idx)
                    finale_rankings.append({
                        'original_idx': idx,
                        'survivor_idx': survivor_idx,
                        'name': season_df.iloc[idx]['celebrity_name'],
                        'final_place': place,
                        'judge_share': judge_share[survivor_idx]
                    })
            week_data['finale_rankings'] = sorted(finale_rankings, key=lambda x: x['final_place'])
        
        # 兼容性：保留单人淘汰的旧字段（如果只有一人被淘汰）
        if len(eliminated_indices) == 1:
            week_data['eliminated_idx'] = eliminated_indices[0]
            week_data['eliminated_idx_in_survivors'] = eliminated_indices_in_survivors[0] if eliminated_indices_in_survivors else None
        else:
            week_data['eliminated_idx'] = None
            week_data['eliminated_idx_in_survivors'] = None
        
        return week_data
    
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
    
    def get_special_weeks_summary(self) -> Dict:
        """获取特殊周次摘要（多人淘汰、无人淘汰、决赛）"""
        if not self.processed_data:
            return {}
        
        summary = {
            'multi_elimination_weeks': [],
            'no_elimination_weeks': [],
            'finale_weeks': []
        }
        
        for season, data in self.processed_data.items():
            for week_num, week_data in data['weeks'].items():
                info = {'season': season, 'week': week_num}
                
                if week_data['is_multi_elimination']:
                    info['n_eliminated'] = week_data['n_eliminated']
                    info['eliminated'] = [data['celebrities'][i] for i in week_data['eliminated_indices']]
                    summary['multi_elimination_weeks'].append(info)
                
                if week_data['is_no_elimination']:
                    info['n_survivors'] = week_data['n_survivors']
                    summary['no_elimination_weeks'].append(info)
                
                if week_data['is_finale']:
                    info['finalists'] = week_data.get('finale_rankings', [])
                    summary['finale_weeks'].append(info)
        
        return summary


if __name__ == "__main__":
    # 测试代码
    loader = DWTSDataLoader("engineered_data.csv")
    
    try:
        loader.load_data()
        
        # 处理所有赛季
        data = loader.process_all_seasons()
        
        print(f"\n成功处理 {len(data)} 个赛季")
        
        # 显示特殊情况摘要
        summary = loader.get_special_weeks_summary()
        
        print(f"\n=== 特殊周次统计 ===")
        print(f"多人淘汰周: {len(summary['multi_elimination_weeks'])}")
        print(f"无人淘汰周: {len(summary['no_elimination_weeks'])}")
        print(f"决赛周: {len(summary['finale_weeks'])}")
        
        # 显示几个多人淘汰的例子
        print(f"\n多人淘汰示例:")
        for info in summary['multi_elimination_weeks'][:5]:
            print(f"  赛季{info['season']} 第{info['week']}周: {info['eliminated']}")
        
        # 显示决赛示例
        print(f"\n决赛示例 (赛季27):")
        if 27 in data:
            s27 = data[27]
            finale_week = s27['max_week']
            if finale_week in s27['weeks']:
                finale_data = s27['weeks'][finale_week]
                if 'finale_rankings' in finale_data:
                    for f in finale_data['finale_rankings']:
                        print(f"  第{f['final_place']}名: {f['name']} (评委份额: {f['judge_share']:.3f})")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
