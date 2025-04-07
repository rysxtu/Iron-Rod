#include <bits/stdc++.h>
using namespace std;

double best = 1;
vector<int> path;

void trade(int cur, double num, vector<vector<double>>& prices, vector<int> cur_path) {
    if (num * prices[cur][3] > best) {
        best = num * prices[cur][3];
        path = cur_path;
    }

    if (cur_path.size() == 4) return;

    for (int i = 0; i <= 3; i++) {
        if (cur == i) continue;
        cur_path.push_back(i);
        trade(i, num*prices[cur][i], prices, cur_path);
        cur_path.pop_back();
    }
}

/*
0 = snowballs
1 = pizzas
2 = nuggets
3 = seashells
*/

int main() {
    vector<vector<double>> adj_matrix = {
        {1, 1.45, 0.52, 0.72},
        {0.7, 1, 0.31, 0.48},
        {1.95, 3.1, 1, 1.49},
        {1.34, 1.98, 0.64, 1}
    };

    for (int i = 0; i < 3; i++) trade(i, adj_matrix[3][i], adj_matrix, {i});

    cout << best << '\n';
    for (int i : path) cout << i << ' ';
}