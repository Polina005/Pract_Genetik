#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

#include "CImg.h"

using namespace std;
using namespace cimg_library;

struct Consts
{
    static int x_value_number;
    static int chromosome_length;
    static int population_size;
    static double crossing_over_probability;
    static double mutation_probability;
    static int generation_number;
    static int selection_pair;
    static int crossover_type;
    static int mutation_type;
    static int init_strategy;
    static int selection_type;
};

int Consts::x_value_number = 5;
int Consts::chromosome_length = 3;
int Consts::population_size = 10;
double Consts::crossing_over_probability = 0.7;
double Consts::mutation_probability = 0.2;
int Consts::generation_number = 50;
int Consts::selection_pair = 2;
int Consts::crossover_type = 1;
int Consts::mutation_type = 1;
int Consts::init_strategy = 1;
int Consts::selection_type = 1;

const vector<int> X_VALUES = { 5, 6, 7, 8, 9 };

struct Specimen
{
    vector<short> chromosome = vector<short>(Consts::chromosome_length, 0);
    int value = 0;
    double fitness = 0.0;
};

double Function(int x)
{
    return pow(x, 3) + 2 * pow(x, 2);
}

int decodeChromosome(const vector<short>& chrom)
{
    int index = 0;
    for (short bit : chrom)
    {
        index = (index << 1) | bit;
    }
    index = index % Consts::x_value_number;
    return X_VALUES[index];
}

void updateSpecimen(Specimen& spec)
{
    spec.value = decodeChromosome(spec.chromosome);
    spec.fitness = Function(spec.value);
}

void Shotgun_generation(vector<Specimen>& current_gen)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, Consts::x_value_number - 1);

    for (int i = 0; i < Consts::population_size; i++)
    {
        int idx = dist(gen);
        int tempIdx = idx;
        for (int j = Consts::chromosome_length - 1; j >= 0; j--)
        {
            current_gen[i].chromosome[j] = tempIdx % 2;
            tempIdx /= 2;
        }
        updateSpecimen(current_gen[i]);
    }
}

void Focusing_generation(vector<Specimen>& current_gen)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(1, Consts::x_value_number - 2);

    for (int i = 0; i < Consts::population_size; i++)
    {
        int idx = dist(gen);
        int tempIdx = idx;
        for (int j = Consts::chromosome_length - 1; j >= 0; j--)
        {
            current_gen[i].chromosome[j] = tempIdx % 2;
            tempIdx /= 2;
        }
        updateSpecimen(current_gen[i]);
    }
}

vector<pair<int, int>> Random_selection()
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, Consts::population_size - 1);
    vector<pair<int, int>> pairs_selected(Consts::selection_pair);

    for (int i = 0; i < Consts::selection_pair; i++)
    {
        int first_memb = dist(gen);
        int second_memb = dist(gen);
        if (first_memb == second_memb)
        {
            second_memb = (second_memb + 1) % Consts::population_size;
        }
        pairs_selected[i] = { first_memb, second_memb };
    }
    return pairs_selected;
}

vector<pair<int, int>> Scaling_selection(vector<Specimen>& current_gen)
{
    random_device rd;
    mt19937 gen(rd());
    vector<pair<int, int>> pairs_selected(Consts::selection_pair);

    vector<double> fitnessValues(Consts::population_size);
    for (int i = 0; i < Consts::population_size; i++)
    {
        fitnessValues[i] = current_gen[i].fitness;
    }

    double maxFitness = *max_element(fitnessValues.begin(), fitnessValues.end());
    double minFitness = *min_element(fitnessValues.begin(), fitnessValues.end());

    if (maxFitness - minFitness < 1e-6)
    {
        for (int i = 0; i < Consts::selection_pair; i++)
        {
            uniform_int_distribution<> dist(0, Consts::population_size - 1);
            int first = dist(gen);
            int second = dist(gen);
            if (first == second) second = (second + 1) % Consts::population_size;
            pairs_selected[i] = { first, second };
        }
        return pairs_selected;
    }

    vector<double> invFitness(Consts::population_size);
    double sumInvFitness = 0;
    for (int i = 0; i < Consts::population_size; i++)
    {
        invFitness[i] = maxFitness - fitnessValues[i] + 0.001;
        sumInvFitness += invFitness[i];
    }

    vector<double> probabilities(Consts::population_size);
    for (int i = 0; i < Consts::population_size; i++)
    {
        probabilities[i] = invFitness[i] / sumInvFitness;
    }
    for (int i = 1; i < Consts::population_size; i++)
    {
        probabilities[i] += probabilities[i - 1];
    }
    probabilities[Consts::population_size - 1] = 1.0;

    uniform_real_distribution<> probDist(0.0, 1.0);

    for (int i = 0; i < Consts::selection_pair; i++)
    {
        double r1 = probDist(gen);
        double r2 = probDist(gen);

        int first_memb = -1, second_memb = -1;
        for (int j = 0; j < Consts::population_size; j++)
        {
            if (first_memb == -1 && r1 <= probabilities[j]) first_memb = j;
            if (second_memb == -1 && r2 <= probabilities[j]) second_memb = j;
        }

        if (first_memb < 0) first_memb = 0;
        if (second_memb < 0) second_memb = 1;
        if (first_memb >= Consts::population_size) first_memb = Consts::population_size - 1;
        if (second_memb >= Consts::population_size) second_memb = Consts::population_size - 2;

        if (first_memb == second_memb)
        {
            second_memb = (second_memb + 1) % Consts::population_size;
        }
        pairs_selected[i] = { first_memb, second_memb };
    }
    return pairs_selected;
}

vector<pair<int, int>> SelectParents(vector<Specimen>& current_gen)
{
    if (Consts::selection_type == 1)
        return Random_selection();
    else
        return Scaling_selection(current_gen);
}

void OnePointCrossover(pair<int, int>& parent, vector<Specimen>& current_gen, vector<Specimen>& new_gen)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> pointDist(1, Consts::chromosome_length - 1);
    int point = pointDist(gen);

    Specimen child1, child2;
    for (int i = 0; i < point; i++)
        child1.chromosome[i] = current_gen[parent.first].chromosome[i];
    for (int i = point; i < Consts::chromosome_length; i++)
        child1.chromosome[i] = current_gen[parent.second].chromosome[i];

    for (int i = 0; i < point; i++)
        child2.chromosome[i] = current_gen[parent.second].chromosome[i];
    for (int i = point; i < Consts::chromosome_length; i++)
        child2.chromosome[i] = current_gen[parent.first].chromosome[i];

    updateSpecimen(child1);
    updateSpecimen(child2);
    new_gen.push_back(child1);
    new_gen.push_back(child2);
}

void TwoPointCrossover(pair<int, int>& parent, vector<Specimen>& current_gen, vector<Specimen>& new_gen)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> pointDist(1, Consts::chromosome_length - 2);
    int point1 = pointDist(gen);
    int point2 = pointDist(gen);
    if (point1 > point2) swap(point1, point2);

    Specimen child1, child2;
    for (int i = 0; i < point1; i++)
    {
        child1.chromosome[i] = current_gen[parent.first].chromosome[i];
        child2.chromosome[i] = current_gen[parent.second].chromosome[i];
    }
    for (int i = point1; i < point2; i++)
    {
        child1.chromosome[i] = current_gen[parent.second].chromosome[i];
        child2.chromosome[i] = current_gen[parent.first].chromosome[i];
    }
    for (int i = point2; i < Consts::chromosome_length; i++)
    {
        child1.chromosome[i] = current_gen[parent.first].chromosome[i];
        child2.chromosome[i] = current_gen[parent.second].chromosome[i];
    }
    updateSpecimen(child1);
    updateSpecimen(child2);
    new_gen.push_back(child1);
    new_gen.push_back(child2);
}

void MultiPointCrossover(pair<int, int>& parent, vector<Specimen>& current_gen, vector<Specimen>& new_gen)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> pointsCountDist(2, Consts::chromosome_length - 1);
    int numPoints = pointsCountDist(gen);

    vector<int> points;
    uniform_int_distribution<> pointDist(1, Consts::chromosome_length - 1);
    for (int p = 0; p < numPoints; p++)
    {
        int pt = pointDist(gen);
        points.push_back(pt);
    }
    sort(points.begin(), points.end());
    points.erase(unique(points.begin(), points.end()), points.end());

    Specimen child1, child2;
    bool fromFirst = true;
    int lastPoint = 0;

    for (int pt : points)
    {
        for (int i = lastPoint; i < pt; i++)
        {
            if (fromFirst)
            {
                child1.chromosome[i] = current_gen[parent.first].chromosome[i];
                child2.chromosome[i] = current_gen[parent.second].chromosome[i];
            }
            else
            {
                child1.chromosome[i] = current_gen[parent.second].chromosome[i];
                child2.chromosome[i] = current_gen[parent.first].chromosome[i];
            }
        }
        fromFirst = !fromFirst;
        lastPoint = pt;
    }

    for (int i = lastPoint; i < Consts::chromosome_length; i++)
    {
        if (fromFirst)
        {
            child1.chromosome[i] = current_gen[parent.first].chromosome[i];
            child2.chromosome[i] = current_gen[parent.second].chromosome[i];
        }
        else
        {
            child1.chromosome[i] = current_gen[parent.second].chromosome[i];
            child2.chromosome[i] = current_gen[parent.first].chromosome[i];
        }
    }

    updateSpecimen(child1);
    updateSpecimen(child2);
    new_gen.push_back(child1);
    new_gen.push_back(child2);
}

void GoldenRatioCrossover(pair<int, int>& parent, vector<Specimen>& current_gen, vector<Specimen>& new_gen)
{
    int point = round(0.618 * Consts::chromosome_length);
    if (point < 1) point = 1;
    if (point >= Consts::chromosome_length) point = Consts::chromosome_length - 1;

    Specimen child1, child2;
    for (int i = 0; i < point; i++)
        child1.chromosome[i] = current_gen[parent.first].chromosome[i];
    for (int i = point; i < Consts::chromosome_length; i++)
        child1.chromosome[i] = current_gen[parent.second].chromosome[i];

    for (int i = 0; i < point; i++)
        child2.chromosome[i] = current_gen[parent.second].chromosome[i];
    for (int i = point; i < Consts::chromosome_length; i++)
        child2.chromosome[i] = current_gen[parent.first].chromosome[i];

    updateSpecimen(child1);
    updateSpecimen(child2);
    new_gen.push_back(child1);
    new_gen.push_back(child2);
}

void Crossover(vector<pair<int, int>>& parents, vector<Specimen>& current_gen, vector<Specimen>& new_gen)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> probDist(0.0, 1.0);

    for (const auto& parent : parents)
    {
        if (parent.first < 0 || parent.first >= Consts::population_size) continue;
        if (parent.second < 0 || parent.second >= Consts::population_size) continue;

        if (probDist(gen) < Consts::crossing_over_probability)
        {
            switch (Consts::crossover_type)
            {
            case 1: OnePointCrossover(const_cast<pair<int, int>&>(parent), current_gen, new_gen); break;
            case 2: TwoPointCrossover(const_cast<pair<int, int>&>(parent), current_gen, new_gen); break;
            case 3: MultiPointCrossover(const_cast<pair<int, int>&>(parent), current_gen, new_gen); break;
            case 4: GoldenRatioCrossover(const_cast<pair<int, int>&>(parent), current_gen, new_gen); break;
            default: OnePointCrossover(const_cast<pair<int, int>&>(parent), current_gen, new_gen); break;
            }
        }
        else
        {
            new_gen.push_back(current_gen[parent.first]);
            new_gen.push_back(current_gen[parent.second]);
        }
    }
}

void PointMutation(vector<Specimen>& new_gen)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> probDist(0.0, 1.0);
    uniform_int_distribution<> bitDist(0, Consts::chromosome_length - 1);

    for (auto& spec : new_gen)
    {
        if (probDist(gen) < Consts::mutation_probability)
        {
            int bitPos = bitDist(gen);
            spec.chromosome[bitPos] = !spec.chromosome[bitPos];
            updateSpecimen(spec);
        }
    }
}

void InversionMutation(vector<Specimen>& new_gen)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> probDist(0.0, 1.0);
    uniform_int_distribution<> posDist(0, Consts::chromosome_length - 1);

    for (auto& spec : new_gen)
    {
        if (probDist(gen) < Consts::mutation_probability)
        {
            int left = posDist(gen);
            int right = posDist(gen);
            if (left > right) swap(left, right);
            if (left == right)
            {
                right = min(right + 1, Consts::chromosome_length - 1);
            }
            reverse(spec.chromosome.begin() + left, spec.chromosome.begin() + right + 1);
            updateSpecimen(spec);
        }
    }
}

void Mutate(vector<Specimen>& new_gen)
{
    if (Consts::mutation_type == 1)
        PointMutation(new_gen);
    else
        InversionMutation(new_gen);
}

Specimen ElitistSelection(vector<Specimen>& new_gen, const vector<Specimen>& current_gen)
{
    if (new_gen.empty()) return Specimen{};

    int best_idx = 0;
    for (int i = 1; i < new_gen.size(); i++)
    {
        if (new_gen[i].fitness < new_gen[best_idx].fitness)
            best_idx = i;
    }

    int best_current = 0;
    for (int i = 1; i < current_gen.size(); i++)
    {
        if (current_gen[i].fitness < current_gen[best_current].fitness)
            best_current = i;
    }

    return (new_gen[best_idx].fitness < current_gen[best_current].fitness)
        ? new_gen[best_idx] : current_gen[best_current];
}

void saveGraph(const vector<double>& bestFitness, const vector<double>& avgFitness, const vector<int>& bestX, const string& filename)
{
    if (bestFitness.empty()) return;

    long long x_min = 1;
    long long x_max = bestFitness.size();

    double y_min = *min_element(bestFitness.begin(), bestFitness.end());
    double y_max = *max_element(bestFitness.begin(), bestFitness.end());

    double avg_min = *min_element(avgFitness.begin(), avgFitness.end());
    double avg_max = *max_element(avgFitness.begin(), avgFitness.end());
    y_min = min(y_min, avg_min);
    y_max = max(y_max, avg_max);

    y_min = max(0.0, y_min - 50);
    y_max = y_max + 50;

    CImg<unsigned char> image(1200, 800, 1, 3, 255);
    const unsigned char black[] = { 0, 0, 0 };
    const unsigned char blue[] = { 0, 0, 255 };
    const unsigned char red[] = { 255, 0, 0 };
    const unsigned char green[] = { 0, 255, 0 };
    const unsigned char gray[] = { 200, 200, 200 };

    int left_margin = 150;
    int bottom_margin = 100;
    int top_margin = 60;
    int right_margin = 50;

    int graph_width = image.width() - left_margin - right_margin;
    int graph_height = image.height() - bottom_margin - top_margin;

    auto transform_x = [&](long long x) -> int {
        double ratio = static_cast<double>(x - x_min) / (x_max - x_min);
        return left_margin + static_cast<int>(ratio * graph_width);
    };

    auto transform_y = [&](double y) -> int {
        return image.height() - bottom_margin - static_cast<int>((y - y_min) * graph_height / (y_max - y_min));
    };

    image.draw_line(left_margin, image.height() - bottom_margin,
        image.width() - right_margin, image.height() - bottom_margin, black);
    image.draw_line(left_margin, image.height() - bottom_margin,
        left_margin, top_margin, black);

    image.draw_text(image.width() / 2 - 80, image.height() - 40, "Generation number", black, 0, 1, 18);
    image.draw_text(40, image.height() / 2 - 50, "f(x)", black, 0, 1, 18);

    image.draw_text(image.width() - 200, 80, "Best value", blue, 0, 1, 14);
    image.draw_text(image.width() - 200, 110, "Average value", red, 0, 1, 14);

    int x_ticks = 10;
    for (int i = 0; i <= x_ticks; i++) {
        long long x_val = x_min + i * (x_max - x_min) / x_ticks;
        int x_pixel = transform_x(x_val);
        image.draw_line(x_pixel, image.height() - bottom_margin,
            x_pixel, image.height() - bottom_margin + 5, black);

        stringstream ss;
        ss << x_val;
        image.draw_text(x_pixel - 40, image.height() - bottom_margin + 15,
            ss.str().c_str(), black, 0, 1, 12);

        image.draw_line(x_pixel, image.height() - bottom_margin,
            x_pixel, top_margin, gray);
    }

    int y_ticks = 10;
    for (int i = 0; i <= y_ticks; i++) {
        double y_val = y_min + i * (y_max - y_min) / y_ticks;
        int y_pixel = transform_y(y_val);

        image.draw_line(left_margin, y_pixel, left_margin - 5, y_pixel, black);

        stringstream ss;
        ss << fixed << setprecision(0) << y_val;
        image.draw_text(left_margin - 120, y_pixel - 8,
            ss.str().c_str(), black, 0, 1, 12);

        image.draw_line(left_margin, y_pixel, image.width() - right_margin, y_pixel, gray);
    }

    for (size_t i = 0; i < bestFitness.size() - 1; i++) {
        int x1 = transform_x(i + 1);
        int y1 = transform_y(bestFitness[i]);
        int x2 = transform_x(i + 2);
        int y2 = transform_y(bestFitness[i + 1]);
        image.draw_line(x1, y1, x2, y2, blue, 2.0f);
    }

    for (size_t i = 0; i < avgFitness.size() - 1; i++) {
        int x1 = transform_x(i + 1);
        int y1 = transform_y(avgFitness[i]);
        int x2 = transform_x(i + 2);
        int y2 = transform_y(avgFitness[i + 1]);
        image.draw_line(x1, y1, x2, y2, red, 2.0f);
    }

    for (size_t i = 0; i < bestFitness.size(); i++) {
        int x = transform_x(i + 1);
        int y = transform_y(bestFitness[i]);
        image.draw_circle(x, y, 4, blue);
        image.draw_circle(x, y, 4, blue, 1.0f, 0U);
    }

    for (size_t i = 0; i < avgFitness.size(); i++) {
        int x = transform_x(i + 1);
        int y = transform_y(avgFitness[i]);
        image.draw_circle(x, y, 3, red);
        image.draw_circle(x, y, 3, red, 1.0f, 0U);
    }

    int lastX = transform_x(bestFitness.size());
    int lastY = transform_y(bestFitness.back());
    image.draw_circle(lastX, lastY, 10, green);
    image.draw_circle(lastX, lastY, 10, green, 1.0f, 0U);

    stringstream bestXstr;
    bestXstr << "Optimal x = " << bestX.back();
    image.draw_text(image.width() - 300, image.height() - 50, bestXstr.str().c_str(), green, 0, 1, 14);

    string title = "Minimization f(x) = x^3 + 2x^2 (interval [5,9])";
    image.draw_text(image.width() / 2 - 200, 20, title.c_str(), black, 0, 1, 20);

    image.save(filename.c_str());
    cout << "Graph saved: " << filename << endl;
}

struct ExperimentResult
{
    string name;
    double bestFitness;
    int bestX;
    vector<double> bestHistory;
    vector<double> avgHistory;
    vector<int> xHistory;
};

ExperimentResult runExperiment(int popSize, double pc, double pm, int generations,
    int initStrategy, int selectionType, int crossoverType, int mutationType,
    const string& expName)
{
    Consts::population_size = popSize;
    Consts::crossing_over_probability = pc;
    Consts::mutation_probability = pm;
    Consts::generation_number = generations;
    Consts::init_strategy = initStrategy;
    Consts::selection_type = selectionType;
    Consts::crossover_type = crossoverType;
    Consts::mutation_type = mutationType;

    vector<Specimen> current_gen(Consts::population_size);

    if (initStrategy == 1)
        Shotgun_generation(current_gen);
    else
        Focusing_generation(current_gen);

    vector<double> bestFitnessHistory;
    vector<double> avgFitnessHistory;
    vector<int> bestXHistory;

    for (int gen = 0; gen < Consts::generation_number; gen++)
    {
        vector<pair<int, int>> parents = SelectParents(current_gen);

        vector<Specimen> new_gen;
        Crossover(parents, current_gen, new_gen);
        Mutate(new_gen);

        if (new_gen.empty())
            new_gen = current_gen;

        Specimen best_individual = ElitistSelection(new_gen, current_gen);

        sort(new_gen.begin(), new_gen.end(), [](const Specimen& a, const Specimen& b) {
            return a.fitness < b.fitness;
            });

        vector<Specimen> next_gen;
        next_gen.push_back(best_individual);

        for (int i = 0; i < Consts::population_size - 1 && i < new_gen.size(); i++)
            next_gen.push_back(new_gen[i]);

        while (next_gen.size() < Consts::population_size)
            next_gen.push_back(best_individual);

        current_gen = next_gen;

        int best_idx = 0;
        double avgFitness = 0;
        for (int i = 0; i < current_gen.size(); i++)
        {
            avgFitness += current_gen[i].fitness;
            if (current_gen[i].fitness < current_gen[best_idx].fitness)
                best_idx = i;
        }
        avgFitness /= current_gen.size();

        bestFitnessHistory.push_back(current_gen[best_idx].fitness);
        avgFitnessHistory.push_back(avgFitness);
        bestXHistory.push_back(current_gen[best_idx].value);
    }

    int final_best_idx = 0;
    for (int i = 1; i < current_gen.size(); i++)
    {
        if (current_gen[i].fitness < current_gen[final_best_idx].fitness)
            final_best_idx = i;
    }

    ExperimentResult res;
    res.name = expName;
    res.bestFitness = current_gen[final_best_idx].fitness;
    res.bestX = current_gen[final_best_idx].value;
    res.bestHistory = bestFitnessHistory;
    res.avgHistory = avgFitnessHistory;
    res.xHistory = bestXHistory;

    return res;
}

void manualInputMode()
{
    cout << "\n--------------" << endl;
    cout << " РУЧНОЙ РЕЖИМ" << endl;
    cout << "--------------" << endl;

    int popSize;
    double pc, pm;
    int generations;
    int initStrategy, selectionType, crossoverType, mutationType;

    string input;

    cout << "Введите размер популяции [10...100] (по умолчанию 10): ";
    getline(cin, input);
    if (input.empty()) popSize = 10;
    else popSize = stoi(input);
    if (popSize < 10) popSize = 10;
    if (popSize > 100) popSize = 100;

    cout << "Введите вероятность кроссинговера [0,1...1,0] (по умолчанию 0,7): ";
    getline(cin, input);
    if (input.empty()) pc = 0.7;
    else pc = stod(input);
    if (pc < 0.1) pc = 0.1;
    if (pc > 1.0) pc = 1.0;

    cout << "Введите вероятность мутации [0,1...1,0] (по умолчанию 0,2): ";
    getline(cin, input);
    if (input.empty()) pm = 0.2;
    else pm = stod(input);
    if (pm < 0.1) pm = 0.1;
    if (pm > 1.0) pm = 1.0;

    cout << "Введите количество поколений [10...1000] (по умолчанию 50): ";
    getline(cin, input);
    if (input.empty()) generations = 50;
    else generations = stoi(input);
    if (generations < 10) generations = 10;
    if (generations > 1000) generations = 1000;

    cout << "Выберите стратегию инициализации (II):" << endl;
    cout << "  1 - дробовик (B)" << endl;
    cout << "  2 - фокусировка (C)" << endl;
    cout << "Введите 1 или 2 (по умолчанию 1): ";
    getline(cin, input);
    if (input.empty()) initStrategy = 1;
    else initStrategy = stoi(input);
    if (initStrategy < 1) initStrategy = 1;
    if (initStrategy > 2) initStrategy = 2;

    cout << "Выберите тип селекции (III):" << endl;
    cout << "  1 - случайная (A)" << endl;
    cout << "  2 - ранговая (C)" << endl;
    cout << "Введите 1 или 2 (по умолчанию 1): ";
    getline(cin, input);
    if (input.empty()) selectionType = 1;
    else selectionType = stoi(input);
    if (selectionType < 1) selectionType = 1;
    if (selectionType > 2) selectionType = 2;

    cout << "Выберите тип кроссинговера (IV):" << endl;
    cout << "  1 - одноточечный (A)" << endl;
    cout << "  2 - двухточечный (B)" << endl;
    cout << "  3 - многоточечный (E)" << endl;
    cout << "  4 - золотое сечение (L)" << endl;
    cout << "Введите 1-4 (по умолчанию 1): ";
    getline(cin, input);
    if (input.empty()) crossoverType = 1;
    else crossoverType = stoi(input);
    if (crossoverType < 1) crossoverType = 1;
    if (crossoverType > 4) crossoverType = 4;

    cout << "Выберите тип мутации (V):" << endl;
    cout << "  1 - точечная (A)" << endl;
    cout << "  2 - инверсия (I)" << endl;
    cout << "Введите 1 или 2 (по умолчанию 1): ";
    getline(cin, input);
    if (input.empty()) mutationType = 1;
    else mutationType = stoi(input);
    if (mutationType < 1) mutationType = 1;
    if (mutationType > 2) mutationType = 2;

    string initStr = (initStrategy == 1) ? "shotgun" : "focus";
    string selStr = (selectionType == 1) ? "random" : "rank";
    string crossStr = (crossoverType == 1) ? "1point" : (crossoverType == 2) ? "2point" : (crossoverType == 3) ? "multipoint" : "golden";
    string mutStr = (mutationType == 1) ? "point" : "inversion";
    string expName = "manual_" + initStr + "_" + selStr + "_" + crossStr + "_" + mutStr;

    cout << "\n--- Запуск эксперимента ---" << endl;
    cout << "Параметры: N=" << popSize << ", pc=" << pc << ", pm=" << pm
        << ", поколений=" << generations
        << ", инициализация=" << ((initStrategy == 1) ? "дробовик" : "фокусировка")
        << ", селекция=" << ((selectionType == 1) ? "случайная" : "ранговая")
        << ", кроссинговер=" << ((crossoverType == 1) ? "1-точ" : (crossoverType == 2) ? "2-точ" : (crossoverType == 3) ? "многоточ" : "зол.сеч")
        << ", мутация=" << ((mutationType == 1) ? "точечная" : "инверсия") << endl;

    ExperimentResult res = runExperiment(popSize, pc, pm, generations, initStrategy, selectionType, crossoverType, mutationType, expName);

    string graphFilename = expName + ".bmp";
    saveGraph(res.bestHistory, res.avgHistory, res.xHistory, graphFilename);

    cout << "\nРЕЗУЛЬТАТ:" << endl;
    cout << "Лучшее x = " << res.bestX << ", f(x) = " << res.bestFitness << endl;

    if (abs(res.bestFitness - 175.0) < 1e-6)
        cout << "Достигнут глобальный минимум (x=5, f=175)!" << endl;
    else
        cout << "Глобальный минимум не достигнут." << endl;
}

int main()
{
    setlocale(LC_ALL, "Russian");
    srand(static_cast<unsigned int>(time(0)));

    cout << "-----------------------------------" << endl;
    cout << " ГЕНЕТИЧЕСКИЙ АЛГОРИТМ МИНИМИЗАЦИИ" << endl;
    cout << "f(x) = x^3 + 2x^2" << endl;
    cout << "x in {5, 6, 7, 8, 9}" << endl;
    cout << "-----------------------------------" << endl;

    cout << "\nВыберите режим работы:" << endl;
    cout << "  1 - Автоматический перебор (32 комбинации II*III*IV*V)" << endl;
    cout << "  2 - Ручной ввод параметров" << endl;
    cout << "Введите 1 или 2: ";

    string modeInput;
    getline(cin, modeInput);
    int mode = (modeInput.empty()) ? 1 : stoi(modeInput);

    if (mode == 2)
    {
        manualInputMode();
        return 0;
    }

    cout << "\nНа графиках отображаются:" << endl;
    cout << "  - СИНЯЯ линия/точки: лучшее значение в поколении" << endl;
    cout << "  - КРАСНАЯ линия/точки: среднее значение по популяции" << endl;
    cout << "  - ЗЕЛЁНАЯ точка: финальный оптимум" << endl;
    cout << "--------------------------------------------------------" << endl;
    cout << "Полный перебор комбинаций:" << endl;
    cout << "  II (B,C) - инициализация: 2 варианта" << endl;
    cout << "  III (A,C) - селекция: 2 варианта" << endl;
    cout << "  IV (A,B,E,L) - кроссинговер: 4 варианта" << endl;
    cout << "  V (A,I) - мутация: 2 варианта" << endl;
    cout << "  ИТОГО: 2 x 2 x 4 x 2 = 32 эксперимента" << endl;
    cout << "--------------------------------------------------------" << endl;

    vector<tuple<int, double, double, int, int, int, int, string>> experiments;
    int expNum = 1;

    for (int init : {1, 2})        
    {
        string initName = (init == 1) ? "shotgun" : "focus";

        for (int sel : {1, 2})       
        {
            string selName = (sel == 1) ? "random" : "rank";

            for (int cross : {1, 2, 3, 4})   
            {
                string crossName = (cross == 1) ? "1point" :
                    (cross == 2) ? "2point" :
                    (cross == 3) ? "multipoint" : "golden";

                for (int mut : {1, 2})        
                {
                    string mutName = (mut == 1) ? "point" : "inversion";
                    string expName = "exp" + to_string(expNum) + "_" + initName + "_" + selName + "_" + crossName + "_" + mutName;
                    experiments.push_back({ 10, 0.7, 0.2, init, sel, cross, mut, expName });
                    expNum++;
                }
            }
        }
    }

    vector<ExperimentResult> results;
    int currentExp = 1;

    for (const auto& exp : experiments)
    {
        int popSize = get<0>(exp);
        double pc = get<1>(exp);
        double pm = get<2>(exp);
        int init = get<3>(exp);
        int sel = get<4>(exp);
        int cross = get<5>(exp);
        int mut = get<6>(exp);
        string expName = get<7>(exp);

        string initStr = (init == 1) ? "дробовик" : "фокусировка";
        string selStr = (sel == 1) ? "случайная" : "ранговая";
        string crossStr = (cross == 1) ? "1-точ" : (cross == 2) ? "2-точ" : (cross == 3) ? "многоточ" : "зол.сеч";
        string mutStr = (mut == 1) ? "точечная" : "инверсия";

        cout << "\n--- Эксперимент " << currentExp << "/" << experiments.size() << " ---" << endl;
        cout << "Параметры: N=10, pc=0.7, pm=0.2, поколений=50" << endl;
        cout << "  инициализация=" << initStr << " (II-" << ((init == 1) ? "B" : "C") << ")" << endl;
        cout << "  селекция=" << selStr << " (III-" << ((sel == 1) ? "A" : "C") << ")" << endl;
        cout << "  кроссинговер=" << crossStr << " (IV-" << ((cross == 1) ? "A" : (cross == 2) ? "B" : (cross == 3) ? "E" : "L") << ")" << endl;
        cout << "  мутация=" << mutStr << " (V-" << ((mut == 1) ? "A" : "I") << ")" << endl;

        ExperimentResult res = runExperiment(popSize, pc, pm, 50, init, sel, cross, mut, expName);

        string graphFilename = expName + ".bmp";
        saveGraph(res.bestHistory, res.avgHistory, res.xHistory, graphFilename);

        cout << "Результат: x=" << res.bestX << ", f(x)=" << res.bestFitness << endl;

        results.push_back(res);
        currentExp++;
    }

    cout << "\n------------------------------" << endl;
    cout << "СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ " << endl;
    cout << "------------------------------" << endl;
    cout << left << setw(50) << "Название эксперимента"
        << setw(12) << "x"
        << setw(12) << "f(x)" << endl;
    cout << "----------------------------------------------------" << endl;

    ExperimentResult* bestResult = nullptr;
    int failedCount = 0;

    for (auto& res : results)
    {
        cout << left << setw(50) << res.name
            << setw(12) << res.bestX
            << setw(12) << res.bestFitness << endl;

        if (abs(res.bestFitness - 175.0) > 1e-6)
            failedCount++;

        if (bestResult == nullptr || res.bestFitness < bestResult->bestFitness)
            bestResult = &res;
    }

    cout << "---------------------------------" << endl;
    cout << "Успешных экспериментов: " << (32 - failedCount) << "/32" << endl;
    cout << "Неудачных: " << failedCount << "/32" << endl;

    if (bestResult != nullptr)
    {
        cout << "\nЛУЧШИЙ РЕЗУЛЬТАТ: " << bestResult->name << endl;
        cout << "x = " << bestResult->bestX << ", f(x) = " << bestResult->bestFitness << endl;

        if (abs(bestResult->bestFitness - 175.0) < 1e-6)
            cout << "Достигнут глобальный минимум (x=5, f=175)" << endl;
    }
    cout << "\nВсе 32 графика сохранены в текущей папке." << endl;
}