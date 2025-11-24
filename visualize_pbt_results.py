"""
PBT 학습 결과 시각화 스크립트
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches


def load_pbt_data(log_dir):
    """PBT 평가 데이터 로드"""
    eval_data_path = os.path.join(log_dir, 'pbt_evaluation_data.json')
    state_path = os.path.join(log_dir, 'pbt_state.json')
    
    if not os.path.exists(eval_data_path):
        raise FileNotFoundError(f"Evaluation data not found: {eval_data_path}")
    
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    state_data = None
    if os.path.exists(state_path):
        with open(state_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
    
    return eval_data, state_data


def plot_learning_curves(eval_data, save_path=None):
    """학습 곡선 플롯"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 색상 맵
    colors = plt.cm.tab10(np.linspace(0, 1, len(eval_data['members'])))
    
    # 1. Mean Distance (모든 개체)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, member in enumerate(eval_data['members']):
        if not member['evaluation_data']:
            continue
        steps = [e['training_steps'] for e in member['evaluation_data']]
        distances = [e['mean_distance'] for e in member['evaluation_data']]
        ax1.plot(steps, distances, alpha=0.6, color=colors[i], 
                label=f"Member {member['member_id']:02d}", linewidth=1.5)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Mean Distance')
    ax1.set_title('Learning Curves: Mean Distance (All Members)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Success Rate (모든 개체)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, member in enumerate(eval_data['members']):
        if not member['evaluation_data']:
            continue
        steps = [e['training_steps'] for e in member['evaluation_data']]
        success_rates = [e['success_rate'] for e in member['evaluation_data']]
        ax2.plot(steps, success_rates, alpha=0.6, color=colors[i], 
                label=f"Member {member['member_id']:02d}", linewidth=1.5)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Learning Curves: Success Rate (All Members)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # 3. Best Distance (최고 성능 추적)
    ax3 = fig.add_subplot(gs[1, 0])
    best_distances = []
    best_steps = []
    best_member_ids = []
    
    all_evaluations = []
    for member in eval_data['members']:
        for eval_record in member['evaluation_data']:
            all_evaluations.append({
                'steps': eval_record['training_steps'],
                'distance': eval_record['best_distance'],
                'member_id': member['member_id']
            })
    
    if all_evaluations:
        all_evaluations.sort(key=lambda x: x['steps'])
        current_best = float('inf')
        for eval_record in all_evaluations:
            if eval_record['distance'] < current_best:
                current_best = eval_record['distance']
                best_distances.append(current_best)
                best_steps.append(eval_record['steps'])
                best_member_ids.append(eval_record['member_id'])
        
        ax3.plot(best_steps, best_distances, 'r-', linewidth=2, label='Best Overall')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Best Distance')
        ax3.set_title('Best Performance Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # 4. Hyperparameter Evolution (주요 하이퍼파라미터)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Learning rate actor 추적
    for i, member in enumerate(eval_data['members']):
        if not member['evaluation_data']:
            continue
        steps = [e['training_steps'] for e in member['evaluation_data']]
        lr_actor = [e['hyperparameters'].get('learning_rate_actor', 0) 
                   for e in member['evaluation_data']]
        ax4.plot(steps, lr_actor, alpha=0.6, color=colors[i], 
                label=f"Member {member['member_id']:02d}", linewidth=1.5)
    
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Learning Rate (Actor)')
    ax4.set_title('Hyperparameter Evolution: Learning Rate Actor')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Generation Statistics
    ax5 = fig.add_subplot(gs[2, 0])
    
    # 세대별 평균 성능
    generation_stats = {}
    for member in eval_data['members']:
        for eval_record in member['evaluation_data']:
            gen = eval_record.get('generation', 0)
            if gen not in generation_stats:
                generation_stats[gen] = []
            generation_stats[gen].append(eval_record['mean_distance'])
    
    if generation_stats:
        generations = sorted(generation_stats.keys())
        mean_distances = [np.mean(generation_stats[g]) for g in generations]
        std_distances = [np.std(generation_stats[g]) for g in generations]
        
        ax5.errorbar(generations, mean_distances, yerr=std_distances, 
                    fmt='o-', capsize=5, capthick=2, linewidth=2)
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Mean Distance')
        ax5.set_title('Average Performance by Generation')
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
    
    # 6. Hyperparameter Changes (Exploit/Explore 이벤트)
    ax6 = fig.add_subplot(gs[2, 1])
    
    # 하이퍼파라미터 변화 횟수
    change_counts = {}
    for member in eval_data['members']:
        for hist in member.get('hyperparameter_history', []):
            gen = hist.get('generation', 0)
            change_counts[gen] = change_counts.get(gen, 0) + 1
    
    if change_counts:
        generations = sorted(change_counts.keys())
        counts = [change_counts[g] for g in generations]
        ax6.bar(generations, counts, alpha=0.7, color='steelblue')
        ax6.set_xlabel('Generation')
        ax6.set_ylabel('Number of Hyperparameter Changes')
        ax6.set_title('Exploit/Explore Events by Generation')
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('PBT Training Results', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_hyperparameter_evolution(eval_data, save_path=None):
    """하이퍼파라미터 진화 플롯"""
    # 주요 하이퍼파라미터 추출
    hyperparams_to_plot = [
        'learning_rate_actor',
        'learning_rate_critic',
        'tau_max',
        'batch_size',
        'noise_std',
        'reward_scale'
    ]
    
    n_params = len(hyperparams_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(eval_data['members'])))
    
    for param_idx, param_name in enumerate(hyperparams_to_plot):
        ax = axes[param_idx]
        
        for i, member in enumerate(eval_data['members']):
            if not member['evaluation_data']:
                continue
            
            steps = [e['training_steps'] for e in member['evaluation_data']]
            values = [e['hyperparameters'].get(param_name, 0) 
                     for e in member['evaluation_data']]
            
            ax.plot(steps, values, alpha=0.6, color=colors[i], 
                   label=f"Member {member['member_id']:02d}", linewidth=1.5)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(param_name)
        ax.set_title(f'Hyperparameter Evolution: {param_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # 로그 스케일이 적합한 경우
        if 'learning_rate' in param_name or 'reward_scale' in param_name:
            ax.set_yscale('log')
    
    plt.suptitle('Hyperparameter Evolution During PBT', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Hyperparameter evolution plot saved to: {save_path}")
    else:
        plt.show()


def plot_member_comparison(eval_data, save_path=None):
    """개체별 비교 플롯"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 최종 성능 비교
    ax1 = axes[0, 0]
    member_ids = [m['member_id'] for m in eval_data['members']]
    best_distances = [m['best_distance'] for m in eval_data['members']]
    colors = ['green' if d == min(best_distances) else 'steelblue' 
             for d in best_distances]
    
    bars = ax1.bar(member_ids, best_distances, color=colors, alpha=0.7)
    ax1.set_xlabel('Member ID')
    ax1.set_ylabel('Best Distance')
    ax1.set_title('Final Best Performance by Member')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # 최종 성능 주석
    for i, (mid, dist) in enumerate(zip(member_ids, best_distances)):
        ax1.text(mid, dist, f'{dist:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 학습 진행도
    ax2 = axes[0, 1]
    training_steps = [m['training_steps'] for m in eval_data['members']]
    ax2.bar(member_ids, training_steps, color='orange', alpha=0.7)
    ax2.set_xlabel('Member ID')
    ax2.set_ylabel('Training Steps')
    ax2.set_title('Training Progress by Member')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 하이퍼파라미터 변화 횟수
    ax3 = axes[1, 0]
    change_counts = [len(m.get('hyperparameter_history', [])) - 1 
                    for m in eval_data['members']]
    ax3.bar(member_ids, change_counts, color='purple', alpha=0.7)
    ax3.set_xlabel('Member ID')
    ax3.set_ylabel('Number of Hyperparameter Changes')
    ax3.set_title('Hyperparameter Changes by Member')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 평가 횟수
    ax4 = axes[1, 1]
    eval_counts = [len(m['evaluation_data']) for m in eval_data['members']]
    ax4.bar(member_ids, eval_counts, color='red', alpha=0.7)
    ax4.set_xlabel('Member ID')
    ax4.set_ylabel('Number of Evaluations')
    ax4.set_title('Evaluation Frequency by Member')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('PBT Member Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Member comparison plot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize PBT training results')
    parser.add_argument('--log-dir', type=str, required=True,
                       help='Directory containing PBT evaluation data')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (default: same as log-dir)')
    parser.add_argument('--plot-type', type=str, default='all',
                       choices=['all', 'learning', 'hyperparams', 'comparison'],
                       help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # 데이터 로드
    eval_data, state_data = load_pbt_data(args.log_dir)
    
    # 출력 디렉토리
    output_dir = args.output_dir if args.output_dir else args.log_dir
    
    print(f"Loaded PBT data from: {args.log_dir}")
    print(f"Generation: {eval_data['generation']}")
    print(f"Total exploits: {eval_data['total_exploits']}")
    print(f"Total explores: {eval_data['total_explores']}")
    print(f"Number of members: {len(eval_data['members'])}")
    
    # 플롯 생성
    if args.plot_type in ['all', 'learning']:
        plot_learning_curves(eval_data, 
                           save_path=os.path.join(output_dir, 'pbt_learning_curves.png'))
    
    if args.plot_type in ['all', 'hyperparams']:
        plot_hyperparameter_evolution(eval_data,
                                    save_path=os.path.join(output_dir, 'pbt_hyperparameter_evolution.png'))
    
    if args.plot_type in ['all', 'comparison']:
        plot_member_comparison(eval_data,
                             save_path=os.path.join(output_dir, 'pbt_member_comparison.png'))
    
    print(f"\nPlots saved to: {output_dir}")


if __name__ == '__main__':
    main()

