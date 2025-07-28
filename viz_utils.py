# viz_utils.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#############################################
# DEMOGRAPHIC VISUALIZATIONS
#############################################

#######
# AGE
#######

def plot_bar_age_groups(data, x_column='AgeGroup', y_column='Frequency', 
                        title="Age Groups Distribution", color='#00008B', 
                        filename=None):
    """
    Creates a bar chart of age groups distribution.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_column], data[y_column], color=color)
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    
    for i, v in enumerate(data[y_column]):
        plt.text(i, v + 5, str(v), ha='center', color='black')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_line_age_by_year(data, x_column='AgeGroup', y_column='Count', 
                          group_column='SurveyID', 
                          title="Age Distribution by Survey Year", 
                          filename=None):
    """
    Creates a line chart showing age groups across different survey years.
    """
    groups = data[group_column].unique()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, group in enumerate(groups):
        group_data = data[data[group_column] == group]
        ax.plot(group_data[x_column], group_data[y_column], 
                marker='o', linewidth=2, label=f'Year {group}',
                color=colors[i % len(colors)])
    
    ax.set_xlabel(x_column, fontsize=12)
    ax.set_ylabel(y_column, fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.legend(title='Survey Year')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if x_column == 'AgeGroup':
        all_age_groups = ['18-24', '25-29', '30-34', '35-39', '40-49', '50-64', '65+']
        ax.set_xticks(range(len(all_age_groups)))
        ax.set_xticklabels(all_age_groups)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

##########
# GENDER
##########

def plot_bar_gender_distribution(data, x_column='StandardizedGender', y_column='Frequency',
                                title="Gender Distribution of Survey Respondents", 
                                color='#4169E1', filename=None):
    """
    Creates a bar chart showing gender distribution.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(data[x_column], data[y_column], color=color)
    plt.title(title)
    plt.xlabel('Gender')
    plt.ylabel('Number of Respondents')
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(data[y_column]):
        plt.text(i, v + 5, str(v), ha='center', color='black')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

###########
# EMPLOYMENT
###########

def plot_pie_binary_response(data, value_column, count_column, 
                           title="Response Distribution", 
                           yes_color='#1E88E5',  
                           no_color='#90CAF9',   
                           other_color='#BBDEFB', 
                           figsize=(8, 6),     
                           filename=None):
    """
    Creates a pie chart for binary (Yes/No) responses with consistent colors and a legend.
    """
    plt.figure(figsize=figsize)
    
    color_map = {'Yes': yes_color, 'No': no_color, 'No Response': other_color}
    
    colors = [color_map.get(val, other_color) for val in data[value_column]]
    
    patches, texts, autotexts = plt.pie(
        data[count_column], 
        labels=None, 
        autopct='%1.1f%%', 
        colors=colors, 
        startangle=90
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    plt.legend(data[value_column], loc='best', fontsize=10)
    
    plt.title(title, fontsize=14)
    plt.axis('equal') 
    
    if filename:
        plt.savefig(filename)
    plt.show()

    

def plot_bar_company_size(data, category_column='CompanySize', count_column='Frequency',
                         title="Company Size Distribution", color='#00408B',  # Darker blue
                         figsize=(10, 6), filename=None):
    """
    Creates a bar chart for company size distribution.
    """
    plt.figure(figsize=figsize)
    
    # Filter out 'No Response' for better visualization
    plot_data = data[data[category_column] != 'No Response'].copy()
    
    bars = plt.bar(plot_data[category_column], plot_data[count_column], color=color)
    plt.title(title, fontsize=14)
    plt.xlabel('Company Size')
    plt.ylabel('Number of Respondents')
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
    
    no_response = data[data[category_column] == 'No Response'][count_column].values
    if len(no_response) > 0:
        total = data[count_column].sum()
        no_response_pct = (no_response[0] / total) * 100
        print(f"'No Response' category: {no_response[0]} respondents ({no_response_pct:.1f}%)")


##############
# DEMOGRAPHICS
##############


def plot_top_countries(data, country_column='Country', count_column='Frequency',
                     percentage_column='Percentage',
                     title="Top Countries of Respondents", color='#00008B',
                     figsize=(13, 6), filename=None):
    """
    Creates a horizontal bar chart of top countries with percentages.
    """
    plt.figure(figsize=figsize)
    
    y_pos = range(len(data))
    bars = plt.barh(y_pos, data[count_column], color=color)
    
    plt.yticks(y_pos, data[country_column])
    
    plt.title(title, fontsize=14)
    plt.xlabel('Number of Respondents')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 20, bar.get_y() + bar.get_height()/2, 
                f'{int(width)} ({data.iloc[i][percentage_column]}%)', 
                ha='left', va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    if filename:
        plt.savefig(filename)
    plt.show()

################
# GENDER_BY_TECH
################

def plot_gender_by_tech(data, year=None, figsize=(12, 8), filename=None):
    
    """
    Creates a grouped bar chart comparing gender distribution in tech vs non-tech companies.
    """
    if year:
        plot_data = data[data['Year'] == year].copy()
        title_year = f" ({year})"
    else:
        plot_data = data.copy()
        plot_data = plot_data.groupby(['Gender', 'CompanyType']).agg({'Count': 'sum'}).reset_index()
        title_year = " (All Years)"
    
    pivot_data = plot_data.pivot(index='Gender', columns='CompanyType', values='Count').fillna(0)
    
    tech_total = pivot_data['Tech'].sum()
    nontech_total = pivot_data['Non-Tech'].sum()
    
    plt.figure(figsize=figsize)
    
    colors = {'Tech': '#1E88E5', 'Non-Tech': '#00408B'}
    
    x = np.arange(len(pivot_data.index))
    width = 0.35
    
    tech_bars = plt.bar(x - width/2, pivot_data['Tech'], width, label='Tech', color=colors['Tech'])
    nontech_bars = plt.bar(x + width/2, pivot_data['Non-Tech'], width, label='Non-Tech', color=colors['Non-Tech'])
    
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title(f'Gender Distribution: Tech vs Non-Tech Companies{title_year}', pad=20) 
    plt.xticks(x, pivot_data.index)
    plt.legend()
    
    def add_labels(bars, total):
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total * 100) if total > 0 else 0
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontsize=9)
    
    add_labels(tech_bars, tech_total)
    add_labels(nontech_bars, nontech_total)
    
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
    plt.show()

    
def plot_gender_trends(data, figsize=(14, 7), filename=None):
    """
    Creates a line chart showing gender distribution trends over survey years.
    """

    trend_data = data.copy()
    
    pivot_data = []
    for year in sorted(trend_data['Year'].unique()):
        for company in sorted(trend_data['CompanyType'].unique()):
            year_company_data = trend_data[(trend_data['Year'] == year) & 
                                          (trend_data['CompanyType'] == company)]
            
            for gender in ['Male', 'Female', 'Other']:
                gender_data = year_company_data[year_company_data['Gender'] == gender]
                if len(gender_data) > 0:
                    percentage = gender_data['Percentage'].values[0]
                else:
                    percentage = 0
                
                pivot_data.append({
                    'Year': year,
                    'CompanyType': company,
                    'Gender': gender,
                    'Percentage': percentage
                })
    
    pivot_df = pd.DataFrame(pivot_data)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    company_types = ['Tech', 'Non-Tech']
    genders = ['Male', 'Female', 'Other']
    colors = {'Male': '#1E88E5', 'Female': '#FFC107', 'Other': '#4CAF50'}
    markers = {'Male': 'o', 'Female': 's', 'Other': '^'}
    
    for i, company in enumerate(company_types):
        ax = axes[i]
        for gender in genders:
            gender_data = pivot_df[(pivot_df['CompanyType'] == company) & 
                                 (pivot_df['Gender'] == gender)]
            
            if not gender_data.empty:
                ax.plot(gender_data['Year'], gender_data['Percentage'], 
                       marker=markers[gender], label=gender, color=colors[gender],
                       linewidth=2, markersize=8)
        
        ax.set_title(f'{company} Companies')
        ax.set_xlabel('Year')
        if i == 0:
            ax.set_ylabel('Percentage (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Gender Distribution Trends by Company Type', fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9) 
    
    if filename:
        plt.savefig(filename)
    plt.show()

#####################
# PREVALENCE ANALYSIS
#####################

def plot_prevalence_with_ci(prevalence_df):
    """
    Plot prevalence rates with confidence intervals for mental health conditions.
    
    Parameters:
    -----------
    prevalence_df : pandas.DataFrame
        DataFrame containing the prevalence data with columns:
        - 'Condition': Name of the mental health condition
        - 'Prevalence': Prevalence rate (as a proportion)
        - 'CI_Lower': Lower bound of confidence interval (as a proportion)
        - 'CI_Upper': Upper bound of confidence interval (as a proportion)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df = prevalence_df.sort_values('Prevalence', ascending=False).reset_index(drop=True)
    
    num_conditions = len(df)
    gray_start = 0.3  
    gray_end = 0.8   
    colors = [cm.Greys(gray_start + i * (gray_end - gray_start) / (num_conditions-1 if num_conditions > 1 else 1)) 
              for i in range(num_conditions)]
    
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['Prevalence'] * 100, align='center', 
                  color=colors, edgecolor='#333333', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Condition'])
    
    max_ci = max(df['CI_Upper'] * 100)
    max_label_space_needed = max_ci + 15 
    
    ax.set_xlim(0, max_label_space_needed)
    
    for i, row in df.iterrows():
        prev = row['Prevalence'] * 100
        ci_lower = row['CI_Lower'] * 100
        ci_upper = row['CI_Upper'] * 100
        
        ax.plot([ci_lower, ci_upper], [i, i], color='black', linewidth=1.5)
        ax.plot([ci_lower, ci_lower], [i-0.1, i+0.1], color='black', linewidth=1.5)
        ax.plot([ci_upper, ci_upper], [i-0.1, i+0.1], color='black', linewidth=1.5)
    
    for i, bar in enumerate(bars):
        row = df.iloc[i]
        prev = row['Prevalence'] * 100
        ci_lower = row['CI_Lower'] * 100
        ci_upper = row['CI_Upper'] * 100
        
        label_position = ci_upper + 1.0 
        
        ax.text(
            label_position, 
            i, 
            f"{prev:.1f}% (95% CI: {ci_lower:.1f}%-{ci_upper:.1f}%)", 
            va='center'
        )
    
    ax.set_xlabel('Prevalence (%)')
    ax.set_title('Prevalence of Mental Health Conditions with 95% Confidence Intervals')
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    if 'Count' in df.columns:
        updated_labels = [f"{cond} (n={count})" for cond, count in zip(df['Condition'], df['Count'])]
        ax.set_yticklabels(updated_labels)
    
    plt.figtext(0.5, 0.01, 
              "Note: Error bars represent 95% confidence intervals.",
              ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
    plt.close(fig)
    
    return fig
    
#################################################
# PREVALENCE ANALYSIS BY GENDER and TECH/NON-TECH
#################################################

def plot_prevalence_comparison(prevalence_data, group_by='Gender', 
                              title="Prevalence of Mental Health Conditions",
                              figsize=(14, 8), filename=None):
    """
    Creates a grouped bar chart comparing prevalence rates across different groups
    (similar to CDC prevalence charts), with improved styling.
    
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 7))
    
    conditions = prevalence_data['Condition'].unique()
    groups = prevalence_data[group_by].unique()
    
    x = np.arange(len(conditions)) 
    width = 0.8 / len(groups)
    
    if group_by == 'Gender':
        colors = ['#8856a7', '#c994c7'] 
    else: 
        colors = ['#1f77b4', '#7bafd2']
    
    for i, group in enumerate(groups):
        group_data = prevalence_data[prevalence_data[group_by] == group]
        
        prevalences = []
        error_bars = []
        
        for condition in conditions:
            condition_data = group_data[group_data['Condition'] == condition]
            if len(condition_data) > 0:
                prev = condition_data['Prevalence'].values[0]
                ci_lower = condition_data['CI_Lower'].values[0]
                ci_upper = condition_data['CI_Upper'].values[0]
                
                prevalences.append(prev)
                error_bars.append([[prev - ci_lower], [ci_upper - prev]])
            else:
                prevalences.append(0)
                error_bars.append([[0], [0]])
        
        error_bars = np.array(error_bars).transpose((1, 0, 2)).squeeze()
        
        bars = plt.bar(x + (i - len(groups)/2 + 0.5) * width, 
                      [p * 100 for p in prevalences],  # Convert to percentage
                      width, label=group, color=colors[i % len(colors)])
        
        for j, (bar, yerr) in enumerate(zip(bars, error_bars.T)):
            height = bar.get_height()
            plt.errorbar(bar.get_x() + bar.get_width()/2, height,
                        yerr=[[yerr[0] * 100], [yerr[1] * 100]],  # Convert to percentage
                        fmt='none', color='black', capsize=3, capthick=1, elinewidth=1)
            
            plt.text(bar.get_x() + bar.get_width()/2, height + max(yerr[1] * 100, 0) + 1.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylabel('Percent', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    
    short_names = []
    for condition in conditions:
        if len(condition) > 15:
            if 'Disorder' in condition:
    
                main_name = condition.split('(')[1].split(',')[0] if '(' in condition else condition.split()[0]
                short_names.append(main_name)
            else:
                short_names.append(condition[:12] + '...')
        else:
            short_names.append(condition)
    
    plt.xticks(x, short_names)
    plt.legend(title=group_by)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    max_height = max([p * 100 + max(e[1] * 100, 0) + 5 for p, e in zip(prevalence_data['Prevalence'], error_bars.T)])
    plt.ylim(0, max_height * 1.2)  # Add 20% more room at the top
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
    
    if any(len(c) > 15 for c in conditions):
        print("Full condition names:")
        for i, condition in enumerate(conditions):
            if len(condition) > 15:
                print(f"{short_names[i]}: {condition}")
                
#############################################
# EDA/DATA CLEANING/UNIQUE VALUES ETC
#############################################

def display_unique_values(df):
    """
    Creates a formatted table showing unique values in each column.
    """

    unique_values_dict = {}
    
    for col in df.columns:
        unique_vals = df[col].unique()
        
        unique_values_dict[col] = {
            'Unique Count': len(unique_vals),
            'Sample Values': str(unique_vals[:5]).replace('\n', ' '),
            'Missing Count': df[col].isna().sum(),
            'Missing %': round(df[col].isna().sum() / len(df) * 100, 1)
        }
    
    unique_df = pd.DataFrame.from_dict(unique_values_dict, orient='index')
    
    unique_df = unique_df.sort_index()
    
    styled_unique = unique_df.style.format({
        'Missing %': '{:.1f}%'
    }).background_gradient(
        cmap='Reds', 
        subset=['Missing %']
    )
    
    return styled_unique

##########################
# MISSING VALUES AND ZEROS
#########################

def plot_missing_and_zeros(df, numeric_cols_only=True):
    """
    Creates a heatmap showing both missing values and zeros in the dataset
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.patches import Patch
    
    if numeric_cols_only:
        data = df.select_dtypes(include=['number'])
    else:
        data = df.copy()
    
    viz_matrix = pd.DataFrame(0, index=data.index, columns=data.columns)
    
    for col in data.columns:
        viz_matrix[col] = np.where(data[col] == 0, 1, viz_matrix[col])
    
    for col in data.columns:
        viz_matrix[col] = np.where(data[col].isnull(), 2, viz_matrix[col])
    
    colors = ['#e6f7ff', '#ffff99', '#ffcccc']
    cmap = sns.color_palette(colors, as_cmap=True)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(viz_matrix, cbar=True, cmap=cmap, yticklabels=False)
    
    legend_elements = [
        Patch(facecolor=colors[0], label='Value Present'),
        Patch(facecolor=colors[1], label='Zero'),
        Patch(facecolor=colors[2], label='Missing Value')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.title('Data Presence Heatmap: Values, Zeros, and Missing Data')
    plt.xlabel('Features')
    plt.ylabel('Observations')
    plt.tight_layout()
    plt.show()

#############################################
# CORRELATION & RELATIONSHIP VISUALIZATIONS
#############################################

def plot_treatment_effectiveness(df):
    """
    Creates visualization comparing work interference with and without treatment.
    Only includes rows with valid data for the specific variables being analyzed in each subplot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # First subplot - Treatment status vs. Work Interference
    # Only use rows with both SoughtTreatment and WorkInterferenceUntreated_Score
    df_subplot1 = df.dropna(subset=['SoughtTreatment', 'WorkInterferenceUntreated_Score'])
    sns.boxplot(x='SoughtTreatment', y='WorkInterferenceUntreated_Score', data=df_subplot1, ax=axes[0])
    axes[0].set_title('Work Interference by Treatment Status\n(Untreated Condition)')
    axes[0].set_ylabel('Work Interference Score\n(0=Never, 3=Often)')
    axes[0].set_xlabel('Sought Treatment')
    
    # Second subplot - Treated vs Untreated interference
    # Only use rows where both interference scores are available
    df_subplot2 = df.dropna(subset=['WorkInterferenceTreated_Score', 'WorkInterferenceUntreated_Score', 'SoughtTreatment'])
    treatment_data = df_subplot2[df_subplot2['SoughtTreatment'] == 'Yes']
    
    if not treatment_data.empty:
        interference_data = pd.DataFrame({
            'Treated': treatment_data['WorkInterferenceTreated_Score'],
            'Untreated': treatment_data['WorkInterferenceUntreated_Score']
        })
        interference_long = interference_data.melt(var_name='Treatment Status', value_name='Interference Score')
        
        sns.boxplot(x='Treatment Status', y='Interference Score', data=interference_long, ax=axes[1])
        axes[1].set_title('Work Interference With vs. Without Treatment\n(For Those Who Sought Treatment)')
    
    plt.tight_layout()
    return fig
    
###########################

def plot_company_support_factors(df):
    """
    Creates visualization showing relationships between company size, 
    mental health benefits, resources, and discussions.
    Only includes rows with valid data for the specific variables being analyzed in each subplot.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5)) 
    
    colors = ['#4682B4', '#7CB9E8', '#B0C4DE']  
    
    size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
    
    df_benefits = df.dropna(subset=['CompanySize', 'MentalHealthBenefits_Binary'])
    
    if not df_benefits.empty:
        benefits_by_size = df_benefits.groupby('CompanySize', observed=True)['MentalHealthBenefits_Binary'].mean() * 100
        
        available_sizes = [size for size in size_order if size in benefits_by_size.index]
        benefits_by_size = benefits_by_size.reindex(available_sizes)
        
        benefits_by_size.plot(kind='bar', color=colors[0], ax=axes[0])
        axes[0].set_title('Mental Health Benefits by Company Size', fontsize=10)
        axes[0].set_ylabel('Percentage with Benefits')
        axes[0].set_ylim(0, 100)
    
    df_resources = df.dropna(subset=['CompanySize', 'MentalHealthResources_Binary'])
    
    if not df_resources.empty:
        resources_by_size = df_resources.groupby('CompanySize', observed=True)['MentalHealthResources_Binary'].mean() * 100
        
        available_sizes = [size for size in size_order if size in resources_by_size.index]
        resources_by_size = resources_by_size.reindex(available_sizes)
        
        resources_by_size.plot(kind='bar', color=colors[1], ax=axes[1])
        axes[1].set_title('Mental Health Resources by Company Size', fontsize=10)
        axes[1].set_ylabel('Percentage with Resources')
        axes[1].set_ylim(0, 100)
    
    df_discuss = df.dropna(subset=['MentalHealthResources_Binary', 'DiscussedMentalHealth_Binary'])
    
    if not df_discuss.empty:
        discussion_by_resources = df_discuss.groupby('MentalHealthResources_Binary', observed=True)['DiscussedMentalHealth_Binary'].mean() * 100
        discussion_by_resources.index = ['No Resources', 'Has Resources']
        discussion_by_resources.plot(kind='bar', color=colors[2], ax=axes[2])
        axes[2].set_title('Mental Health Discussions by Resource Availability', fontsize=10)
        axes[2].set_ylabel('Percentage Discussing Mental Health')
        axes[2].set_ylim(0, 100)
    
    plt.tight_layout()
    return fig
    
    
#############################

def plot_remote_work_relationships(df):
    """
    Creates visualization showing relationships between remote work status,
    mental health conditions, and company size.
    Only includes rows with valid data for the specific variables being analyzed in each subplot.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    colors = ['#6a5acd', '#20b2aa'] 
    
    # Remote work vs. mental health prevalence - pairwise deletion
    df_remote_mh = df.dropna(subset=['RemoteWork', 'CurrentMentalHealth_Binary'])
    
    if len(df_remote_mh) > 0:
        mh_by_remote = df_remote_mh.groupby('RemoteWork')['CurrentMentalHealth_Binary'].mean() * 100
        mh_by_remote.plot(kind='bar', color=colors[0], ax=axes[0])
        axes[0].set_title('Current Mental Health Condition\nby Remote Work Status')
        axes[0].set_ylabel('Percentage with Condition')
        axes[0].set_ylim(0, 100)
    else:
        axes[0].text(0.5, 0.5, "Insufficient data with both Remote Work\nand Mental Health information", 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    
    # Remote work across company sizes - pairwise deletion
    df_remote_size = df.dropna(subset=['CompanySize', 'RemoteWork_Binary'])
    
    if len(df_remote_size) > 0:
        size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
        
        remote_by_size = df_remote_size.groupby('CompanySize')['RemoteWork_Binary'].mean() * 100
        
        available_sizes = [size for size in size_order if size in remote_by_size.index]
        remote_by_size = remote_by_size.reindex(available_sizes)
        
        remote_by_size.plot(kind='bar', color=colors[1], ax=axes[1])
        axes[1].set_title('Remote Work Prevalence by Company Size')
        axes[1].set_ylabel('Percentage Working Remotely')
        axes[1].set_ylim(0, 100)
    else:
        axes[1].text(0.5, 0.5, "Insufficient data for remote work by company size", 
                   ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    plt.tight_layout()
    return fig

#################################

def plot_diagnosis_treatment_interference(df):
    """
    Creates visualization showing relationships between diagnosis status,
    treatment-seeking behavior, and work interference.
    Only includes rows with valid data for the specific variables being analyzed in each subplot.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    colors = ['#4682B4', '#A2C4E0'] 
    
    df_diag_treat = df.dropna(subset=['DiagnosedMentalHealth_Binary', 'SoughtTreatment_Binary'])
    
    if not df_diag_treat.empty:
        treatment_by_diagnosis = df_diag_treat.groupby('DiagnosedMentalHealth_Binary')['SoughtTreatment_Binary'].mean() * 100
        treatment_by_diagnosis.index = ['Not Diagnosed', 'Diagnosed']
        treatment_by_diagnosis.plot(kind='bar', color=colors[0], ax=axes[0])
        axes[0].set_title('Treatment-Seeking by Diagnosis Status')
        axes[0].set_ylabel('Percentage Seeking Treatment')
        axes[0].set_ylim(0, 100)
    else:
        axes[0].text(0.5, 0.5, "Insufficient data for diagnosis vs. treatment", 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    
    # Work interference by diagnosis - pairwise deletion for both interference scores
    df_diag_interf = df.dropna(subset=['DiagnosedMentalHealth_Binary', 'WorkInterferenceTreated_Score', 
                                      'WorkInterferenceUntreated_Score'])
    
    if not df_diag_interf.empty:
        interference_by_diagnosis = df_diag_interf.groupby('DiagnosedMentalHealth_Binary')[
            ['WorkInterferenceTreated_Score', 'WorkInterferenceUntreated_Score']].mean()
        interference_by_diagnosis.index = ['Not Diagnosed', 'Diagnosed']
        interference_by_diagnosis.plot(kind='bar', color=colors, ax=axes[1])
        axes[1].set_title('Work Interference by Diagnosis Status')
        axes[1].set_ylabel('Average Interference Score\n(0=Never, 3=Often)')
        axes[1].legend(['Treated', 'Untreated'])
    
    plt.tight_layout()
    return fig

##########################


def plot_age_mental_health(df):
    """
    Creates visualization showing relationships between age groups,
    mental health conditions, and treatment-seeking behavior.
    Only includes rows with valid data for the specific variables being analyzed in each subplot.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing age and mental health data
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object containing the visualization
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    colors = ['#6a51a3', '#9e9ac8'] 
    
    age_col = 'AgeGroup_Capped' if 'AgeGroup_Capped' in df.columns else 'AgeGroup'
    
    df_age_mh = df.dropna(subset=[age_col, 'CurrentMentalHealth_Binary'])
    
    if not df_age_mh.empty:
        mh_by_age = df_age_mh.groupby(age_col, observed=True)['CurrentMentalHealth_Binary'].mean() * 100
        mh_by_age.plot(kind='bar', color=colors[0], ax=axes[0])
        axes[0].set_title('Current Mental Health Condition by Age Group')
        axes[0].set_ylabel('Percentage with Condition')
        axes[0].set_ylim(0, 100)
    else:
        axes[0].text(0.5, 0.5, "Insufficient data for age vs. mental health", 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    
    df_age_treat = df.dropna(subset=[age_col, 'SoughtTreatment_Binary'])
    
    if not df_age_treat.empty:
        treatment_by_age = df_age_treat.groupby(age_col, observed=True)['SoughtTreatment_Binary'].mean() * 100
        treatment_by_age.plot(kind='bar', color=colors[1], ax=axes[1])
        axes[1].set_title('Treatment-Seeking by Age Group')
        axes[1].set_ylabel('Percentage Seeking Treatment')
        axes[1].set_ylim(0, 100)
    else:
        axes[1].text(0.5, 0.5, "Insufficient data for age vs. treatment", 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    plt.tight_layout()
    return fig