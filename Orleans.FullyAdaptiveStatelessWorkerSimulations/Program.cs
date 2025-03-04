using ScottPlot;
using System.Drawing;
using System.Windows.Forms;

int maxWorkers = 50;
double messageRateSeed = 100;
double simulationDuration = 60;
double workerProcessingTime = 0.1;
double simulationTimeStep = 0.1;
double simulationRemovalPeriod = 0.5;

Mode mode = Mode.Run;
ArrivalPattern pattern = ArrivalPattern.All;

Dictionary<ArrivalPattern, (double Kp, double Ki, double Kd)> finalPidParams = [];

if (pattern != ArrivalPattern.All)
{
    if (pattern == ArrivalPattern.Poisson && mode == Mode.Tune)
    {
        throw new InvalidOperationException("The Poisson arrival pattern is not meant to be used in tune mode.");
    }

    RunApp(pattern, finalPidParams, mode);
}
else
{
    var exceptList = new List<ArrivalPattern>()
    {
        ArrivalPattern.All
    };

    if (mode == Mode.Tune)
    {
        exceptList.Add(ArrivalPattern.Poisson);
    }

    foreach (var _pattern in Enum.GetValues<ArrivalPattern>().Cast<ArrivalPattern>().Except(exceptList))
    {
        RunApp(_pattern, finalPidParams, mode);
    }
}

if (mode == Mode.Tune)
{
    if (finalPidParams.Count > 0)
    {
        double avgKp = finalPidParams.Values.Average(k => k.Kp);
        double avgKi = finalPidParams.Values.Average(k => k.Ki);
        double avgKd = finalPidParams.Values.Average(k => k.Kd);

        Console.WriteLine("\nAverage Optimized PID Values Across All Patterns:");
        Console.WriteLine($"Avg Kp = {avgKp:F3}, Avg Ki = {avgKi:F3}, Avg Kd = {avgKd:F3}");
    }

    Console.ReadKey();
}

void RunApp(ArrivalPattern pattern, Dictionary<ArrivalPattern, (double Kp, double Ki, double Kd)> finalKValues, Mode mode)
{
    var messageArrivalTimes = GenerateMessageArrivalTimes(messageRateSeed, simulationDuration, pattern);

    var pidParams = mode == Mode.Run ?
        (Kp: 0.433, Ki: 0.468, Kd: 0.480) : // The already tuned params
        TunePIDWithGeneticAlgorithm(messageArrivalTimes).First();

    Console.WriteLine($"PID Values for {pattern}: Kp = {pidParams.Kp}, Ki = {pidParams.Ki}, Kd = {pidParams.Kd}");

    finalKValues[pattern] = pidParams;

    var originalSimulation = new Simulation(workerProcessingTime, simulationDuration, maxWorkers, false, simulationTimeStep, simulationRemovalPeriod, messageArrivalTimes);
    originalSimulation.Run();

    var dynamicSimulation = new Simulation(workerProcessingTime, simulationDuration, maxWorkers, true, simulationTimeStep, simulationRemovalPeriod, messageArrivalTimes)
    {
        Kp = pidParams.Kp,
        Ki = pidParams.Ki,
        Kd = pidParams.Kd
    };
    dynamicSimulation.Run();

    Console.WriteLine("\nPerformance Comparison:");
    Console.WriteLine($"Static: Avg Workers = {originalSimulation.GetAverageWorkerCount()}, Avg Queue Length = {originalSimulation.QueueLengthData.Average(d => d.Value)}");
    Console.WriteLine($"Dynamic: Avg Workers = {dynamicSimulation.GetAverageWorkerCount()}, Avg Queue Length = {dynamicSimulation.QueueLengthData.Average(d => d.Value)}");

    if (mode == Mode.Run)
    {
        Application.Run(CombinedPlots(originalSimulation, dynamicSimulation, pattern));
    }
}

List<(double Kp, double Ki, double Kd)> TunePIDWithGeneticAlgorithm(List<double> messageArrivalTimes)
{
    int populationSize = 20;
    int generations = 50;
    double mutationRate = 0.1;

    // Initialize population
    var population = new List<(double Kp, double Ki, double Kd)>();

    for (int i = 0; i < populationSize; i++)
    {
        population.Add((
            Math.Abs(Random.Shared.NextDouble()),
            Math.Abs(Random.Shared.NextDouble()),
            Math.Abs(Random.Shared.NextDouble())
        ));
    }

    // Evolve population
    for (int gen = 0; gen < generations; gen++)
    {
        // Evaluate fitness
        var fitness = population.Select(p => CalculateFitness(p.Kp, p.Ki, p.Kd, messageArrivalTimes)).ToList();

        // Select best individuals
        var bestIndividuals = population.Zip(fitness, (p, f) => (p, f))
            .OrderBy(x => x.f)
            .Take(populationSize / 2)
            .Select(x => x.p)
            .ToList();

        // Crossover and mutate
        var newPopulation = new List<(double Kp, double Ki, double Kd)>();
        while (newPopulation.Count < populationSize)
        {
            var parent1 = bestIndividuals[Random.Shared.Next(bestIndividuals.Count)];
            var parent2 = bestIndividuals[Random.Shared.Next(bestIndividuals.Count)];

            (var childKp, var childKi, var childKd) = (
                Math.Abs((parent1.Kp + parent2.Kp) / 2),
                Math.Abs((parent1.Ki + parent2.Ki) / 2),
                Math.Abs((parent1.Kd + parent2.Kd) / 2)
            );

            // Mutate
            if (Random.Shared.NextDouble() < mutationRate)
            {
                childKp = Math.Abs(childKp + (Random.Shared.NextDouble() - 0.5) * 0.1);
                childKi = Math.Abs(childKi + (Random.Shared.NextDouble() - 0.5) * 0.1);
                childKd = Math.Abs(childKd + (Random.Shared.NextDouble() - 0.5) * 0.1);
            }

            newPopulation.Add((childKp, childKi, childKd));
        }

        population = newPopulation;
    }

    // Calculate fitness for each individual and store the results
    var fitnessValues = population
        .Select(p => (p, CalculateFitness(p.Kp, p.Ki, p.Kd, messageArrivalTimes)))
        .ToList();

    // Find the best fitness
    double bestFitness = fitnessValues.Min(x => x.Item2);

    // Find all solutions with fitness close to the best (using a tolerance)
    const double Tolerance = 1e-6;

    var bestSolutions = fitnessValues
        .Where(x => Math.Abs(x.Item2 - bestFitness) < Tolerance)
        .Select(x => x.p)
        .ToList();

    if (bestSolutions.Count == 0)
    {
        Console.WriteLine("No valid solutions found. Returning default PID values.");
        return [(1.2, 0.4, 0.3)]; // Some defaults
    }

    return bestSolutions;
}

double CalculateFitness(double kp, double ki, double kd, List<double> messageArrivalTimes)
{
    try
    {
        var simulation = new Simulation(workerProcessingTime, simulationDuration, maxWorkers, true, simulationTimeStep, simulationRemovalPeriod, messageArrivalTimes)
        {
            Kp = kp,
            Ki = ki,
            Kd = kd
        };

        simulation.Run();

        double avgWaitingCount = simulation.AvgWaitingCountData.Average(d => d.Value);
        double avgQueueLength = simulation.QueueLengthData.Average(d => d.Value);
        double overshoot = Math.Max(0, simulation.QueueLengthData.Max(d => d.Value) - avgQueueLength);
        double settlingTime = simulation.QueueLengthData.Last().Time;

        double fitness = avgWaitingCount + avgQueueLength + overshoot + settlingTime;
        Console.WriteLine($"KP: {kp:F3}, KI: {ki:F3}, KD: {kd:F3}, Fitness: {fitness:F3}");

        return fitness;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error in CalculateFitness: {ex.Message}");
        return double.MaxValue; // We eturn a high fitness value for invalid parameters.
    }
}

static List<double> GenerateMessageArrivalTimes(double messageRateSeed, double simulationDuration, ArrivalPattern pattern)
{
    var arrivalTimes = new List<double>();

    switch (pattern)
    {
        case ArrivalPattern.Constant:
            {
                double interval = 1 / messageRateSeed;
                for (double time = interval; time < simulationDuration; time += interval)
                {
                    arrivalTimes.Add(time);
                }
            }
            break;

        case ArrivalPattern.Periodic:
            {
                double nextArrivalTime = 0;
                double lowRateStart1 = simulationDuration / 4;
                double lowRateEnd1 = simulationDuration / 2;
                double lowRateStart2 = 3 * simulationDuration / 4;
                double lowRateEnd2 = simulationDuration;
                double lowRateMultiplier = 0.1;

                while (nextArrivalTime < simulationDuration)
                {
                    double dynamicRate;
                    if ((nextArrivalTime > lowRateStart1 && nextArrivalTime < lowRateEnd1) ||
                        (nextArrivalTime > lowRateStart2 && nextArrivalTime < lowRateEnd2))
                    {
                        dynamicRate = messageRateSeed * lowRateMultiplier * (1 + Random.Shared.NextDouble() - 0.5);
                    }
                    else
                    {
                        dynamicRate = messageRateSeed * (1 + Random.Shared.NextDouble() - 0.5);
                    }

                    nextArrivalTime += 1 / dynamicRate;
                    if (nextArrivalTime > 0 && nextArrivalTime < simulationDuration)
                    {
                        arrivalTimes.Add(nextArrivalTime);
                    }
                }
            }
            break;

        case ArrivalPattern.Ramp:
            {
                double rampUpDuration = 0.5 * simulationDuration;
                double rampDownDuration = 0.5 * simulationDuration;
                double rampUpRate = messageRateSeed / rampUpDuration;
                double rampDownRate = messageRateSeed / rampDownDuration;
                double nextRampTime = 0;
                double currentRate = 0;

                while (nextRampTime < simulationDuration)
                {
                    if (nextRampTime < rampUpDuration)
                    {
                        currentRate = rampUpRate * nextRampTime;
                    }
                    else
                    {
                        currentRate = messageRateSeed - rampDownRate * (nextRampTime - rampUpDuration);
                    }

                    if (currentRate <= 0)
                    {
                        nextRampTime += 1;
                        continue;
                    }

                    nextRampTime += 1 / currentRate;
                    if (nextRampTime > 0 && nextRampTime < simulationDuration)
                    {
                        arrivalTimes.Add(nextRampTime);
                    }
                }
            }
            break;

        case ArrivalPattern.Spike:
            {
                double spikeInterval = 10;
                double spikeDuration = 0.05;
                int spikeMessageCount = (int)(messageRateSeed * spikeDuration);

                for (double time = spikeInterval; time < simulationDuration; time += spikeInterval)
                {
                    for (int i = 0; i < spikeMessageCount; i++)
                    {
                        arrivalTimes.Add(time + Random.Shared.NextDouble() * spikeDuration);
                    }
                }
            }
            break;

        case ArrivalPattern.SpikeAndDecay:
            {
                double nextSpikeTime = 0;
                double baseRate = messageRateSeed * 0.1; // Base message rate between spikes
                double spikeMultiplier = 5.0; // How much the rate increases during a spike
                double decayRate = 0.9; // Rate at which the spike decays (e.g., 90% of previous rate)
                double minSpikeInterval = 5.0; // Minimum time between spikes
                double maxSpikeInterval = 15.0; // Maximum time between spikes

                while (nextSpikeTime < simulationDuration)
                {
                    // Generate a spike
                    double spikeRate = messageRateSeed * spikeMultiplier;
                    double spikeDuration = 1.0; // Duration of the spike
                    double spikeEndTime = nextSpikeTime + spikeDuration;

                    // Add messages during the spike
                    while (nextSpikeTime < spikeEndTime && nextSpikeTime < simulationDuration)
                    {
                        nextSpikeTime += 1 / spikeRate;
                        if (nextSpikeTime > 0 && nextSpikeTime < simulationDuration)
                        {
                            arrivalTimes.Add(nextSpikeTime);
                        }
                    }

                    // Decay the spike rate gradually
                    while (spikeRate > baseRate && nextSpikeTime < simulationDuration)
                    {
                        spikeRate *= decayRate; // Reduce the spike rate
                        double nextMessageTime = nextSpikeTime + 1 / spikeRate;
                        if (nextMessageTime > 0 && nextMessageTime < simulationDuration)
                        {
                            arrivalTimes.Add(nextMessageTime);
                        }
                        nextSpikeTime = nextMessageTime;
                    }

                    // Add a random interval before the next spike
                    double nextSpikeInterval = minSpikeInterval + (maxSpikeInterval - minSpikeInterval) * Random.Shared.NextDouble();
                    nextSpikeTime += nextSpikeInterval;
                }
            }
            break;

        case ArrivalPattern.Chaotic:
            {
                double baseRate = messageRateSeed * 0.1;
                double burstProbability = 0.1;
                double burstDuration = 1;
                int burstMessageCount = (int)(baseRate * burstDuration * 2);
                double nextTime = 0;

                while (nextTime < simulationDuration)
                {
                    double dynamicRate = baseRate;
                    if (Random.Shared.NextDouble() < burstProbability)
                    {
                        dynamicRate = messageRateSeed * Random.Shared.NextDouble() * 2;
                        for (int i = 0; i < burstMessageCount; i++)
                        {
                            arrivalTimes.Add(nextTime + Random.Shared.NextDouble() * burstDuration);
                        }

                        nextTime += burstDuration;
                        burstDuration += burstDuration / 2;
                    }
                    else
                    {
                        nextTime += 1 / dynamicRate;
                        if (nextTime > 0 && nextTime < simulationDuration)
                        {
                            arrivalTimes.Add(nextTime);
                        }
                    }
                }
            }
            break;

        case ArrivalPattern.Poisson:
            {
                // Arrival times using the Poisson distribution.
                // Δt = -ln(U)/λ where U ~ Uniform(0,1) and λ = messageRateSeed.

                double nextArrivalTime = 0;
                while (nextArrivalTime < simulationDuration)
                {
                    double u = Random.Shared.NextDouble();
                    double delta = -Math.Log(u) / messageRateSeed;

                    nextArrivalTime += delta;

                    if (nextArrivalTime > 0 && nextArrivalTime < simulationDuration)
                    {
                        arrivalTimes.Add(nextArrivalTime);
                    }
                }
            }
            break;
    }

    return arrivalTimes;
}

static Form CombinedPlots(Simulation _static, Simulation _dynamic, ArrivalPattern pattern)
{
    var form = new Form
    {
        Text = $"Simulation Results ({pattern})",
        Size = new Size(1600, 1200),
        WindowState = FormWindowState.Maximized
    };

    var tableLayoutPanel = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4 };

    tableLayoutPanel.RowCount = 4;
    for (int i = 0; i < tableLayoutPanel.RowCount; i++)
    {
        tableLayoutPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 25f));
    }

    form.Controls.Add(tableLayoutPanel);

    AddMessageRatePlot(tableLayoutPanel, _static.MessageRateData, "Message Rate", "Time", "Messages");
    AddPlot(tableLayoutPanel, _static.WorkerCountData, _dynamic.WorkerCountData, "Worker Count", "Time", "Workers", _static.GetAverageWorkerCount(), _dynamic.GetAverageWorkerCount());
    AddPlot(tableLayoutPanel, _static.QueueLengthData, _dynamic.QueueLengthData, "Queue Length", "Time", "Queue");
    AddPlot(tableLayoutPanel, _static.AvgWaitingCountData, _dynamic.AvgWaitingCountData, "Avg Waiting Count", "Time", "Waiting Count");

    return form;

    static void AddMessageRatePlot(TableLayoutPanel panel, List<(double Time, double Value)> data, string title, string xLabel, string yLabel)
    {
        var plot = new FormsPlot { Dock = DockStyle.Fill };
        panel.Controls.Add(plot);
        plot.Plot.AddScatter(data.Select(d => d.Time).ToArray(), data.Select(d => d.Value).ToArray(), color: Color.DarkGreen);
        plot.Plot.Title(title);
        plot.Plot.XLabel(xLabel);
        plot.Plot.YLabel(yLabel);
        plot.Plot.AxisAuto();
        plot.Refresh();
    }

    static void AddPlot(TableLayoutPanel panel, List<(double Time, double Value)> dataOriginal, List<(double Time, double Value)> dataHC, string title, string xLabel, string yLabel, double avgOriginal = double.NaN, double avgHC = double.NaN)
    {
        var plot = new FormsPlot { Dock = DockStyle.Fill };
        panel.Controls.Add(plot);
        plot.Plot.AddScatter(dataOriginal.Select(d => d.Time).ToArray(), dataOriginal.Select(d => d.Value).ToArray(), label: "Static", color: Color.Blue);
        plot.Plot.AddScatter(dataHC.Select(d => d.Time).ToArray(), dataHC.Select(d => d.Value).ToArray(), label: "Dynamic", color: Color.Red);

        if (!double.IsNaN(avgOriginal))
        {
            plot.Plot.AddHorizontalLine(avgOriginal, color: Color.Blue, style: LineStyle.Dash, label: "Avg Workers (Static)");
        }
        if (!double.IsNaN(avgHC))
        {
            plot.Plot.AddHorizontalLine(avgHC, color: Color.Red, style: LineStyle.Dash, label: "Avg Workers (Dynamic)");
        }

        plot.Plot.Title(title);
        plot.Plot.XLabel(xLabel);
        plot.Plot.YLabel(yLabel);
        plot.Plot.Legend();
        plot.Refresh();
    }
}

enum Mode
{
    Run,
    Tune
}

enum ArrivalPattern
{
    All,
    Constant,
    Periodic,
    Ramp,
    Spike,
    SpikeAndDecay,
    Chaotic,
    Poisson
}

public class Message
{
    public double ArrivalTime { get; set; }
}

public class Worker
{
    public bool IsInactive { get; set; }
    public int WaitingCount { get; set; }
    public double ProcessingTime { get; set; }
    public double FinishTime { get; set; }
}

public class Simulation(double workerProcessingTime, double simulationDuration, int maxWorkers, bool isDynamic, double simulationTimeStep, double simulationRemovalPeriod, List<double> messageArrivalTimes)
{
    public List<(double Time, double Value)> WorkerCountData = [];
    public List<(double Time, double Value)> QueueLengthData = [];
    public List<(double Time, double Value)> AvgWaitingCountData = [];
    public List<(double Time, double Value)> MessageRateData = [];

    private int _queueLength = 0;
    private double _currentSimulationTime = 0;
    private double _lastWorkerRemovalSimulationTime = 0;
    private DateTime _lastWorkerRemovalTime = DateTime.MinValue;

    public double Kp { get; set; }
    public double Ki { get; set; }
    public double Kd { get; set; }

    private double _previousError = 0;
    private double _integralTerm = 0;
    private double _timeBelowZero = 0;

    private readonly int _maxWorkers = maxWorkers;
    private readonly int _workerRemovalBackoffMs = 5 * (int)simulationRemovalPeriod;
    private readonly double _workerProcessingTime = workerProcessingTime;
    private readonly double _simulationDuration = simulationDuration;
    private readonly double _simulationTimeStep = simulationTimeStep;
    private readonly double _timeBelowZeroHysteresis = 10 * simulationTimeStep;
    private readonly double _simulationRemovalPeriod = simulationRemovalPeriod;

    private readonly List<Worker> _workers = [];
    private readonly Queue<Message> _messageQueue = new();
    private readonly List<double> _messageArrivalTimes = messageArrivalTimes;

    public void Run()
    {
        GenerateWorkloadFromArrivalTimes();
        while (_currentSimulationTime < _simulationDuration)
        {
            int messageCount = ProcessMessages();
            if (isDynamic && _currentSimulationTime >= _lastWorkerRemovalSimulationTime + _simulationRemovalPeriod)
            {
                TryRemoveWorkers();
                _lastWorkerRemovalSimulationTime = _currentSimulationTime;
            }

            SimulateWorkerProcessing();
            TrackMetrics();
            TrackMessageRate(messageCount);

            _currentSimulationTime += _simulationTimeStep;
        }
    }

    private void GenerateWorkloadFromArrivalTimes()
    {
        foreach (var arrivalTime in _messageArrivalTimes)
        {
            _messageQueue.Enqueue(new Message { ArrivalTime = arrivalTime });
        }
    }

    private int ProcessMessages()
    {
        int messageCount = 0;
        while (_messageQueue.Count > 0 && _messageQueue.Peek().ArrivalTime <= _currentSimulationTime)
        {
            var message = _messageQueue.Dequeue();
            _queueLength++;
            messageCount++;

            var worker = _workers.FirstOrDefault(w => w.IsInactive);
            if (worker == null)
            {
                if (_workers.Count < _maxWorkers)
                {
                    worker = CreateWorker();
                }
                else
                {
                    worker = _workers.OrderBy(w => w.WaitingCount).First();
                }
            }

            worker.WaitingCount++;
            worker.FinishTime = _currentSimulationTime + worker.ProcessingTime * (1 + (Random.Shared.NextDouble() - 0.5));
            worker.IsInactive = false;
        }
        return messageCount;
    }

    private void TryRemoveWorkers()
    {
        double currentAvgWaitingCount = _workers.Count > 0 ? _workers.Average(w => w.WaitingCount) : 0;
        double error = -1 * currentAvgWaitingCount;

        _integralTerm += error;
        double derivative = error - _previousError;
        _previousError = error;

        double controlSignal = Kp * error + Ki * _integralTerm + Kd * derivative;

        if ((DateTime.Now - _lastWorkerRemovalTime).TotalMilliseconds > _workerRemovalBackoffMs)
        {
            if (controlSignal < 0)
            {
                _timeBelowZero += simulationTimeStep;
                if (_timeBelowZero > _timeBelowZeroHysteresis)
                {
                    var workerToRemove = _workers.LastOrDefault(w => w.IsInactive);
                    if (workerToRemove != null)
                    {
                        _workers.Remove(workerToRemove);
                        _lastWorkerRemovalTime = DateTime.UtcNow;
                    }

                    _timeBelowZero = 0;
                }
            }
            else
            {
                _timeBelowZero = 0;
            }
        }
    }

    private Worker CreateWorker()
    {
        var newWorker = new Worker
        {
            ProcessingTime = _workerProcessingTime * (1 + Random.Shared.NextDouble() * 0.5),
        };
        _workers.Add(newWorker);
        return newWorker;
    }

    private void SimulateWorkerProcessing()
    {
        foreach (var worker in _workers)
        {
            if (worker.FinishTime <= _currentSimulationTime)
            {
                if (_queueLength > 0)
                {
                    _queueLength = Math.Max(0, _queueLength - 1);
                }
                worker.IsInactive = true;
                worker.WaitingCount = 0;
            }
        }
    }

    private void TrackMetrics()
    {
        WorkerCountData.Add((_currentSimulationTime, _workers.Count));
        QueueLengthData.Add((_currentSimulationTime, _queueLength));
        var avgWaitingCount = _workers.Count > 0 ? _workers.Average(w => w.WaitingCount) : 0;
        AvgWaitingCountData.Add((_currentSimulationTime, avgWaitingCount));
    }

    private void TrackMessageRate(int messageCount)
    {
        double messageRate = messageCount / _simulationTimeStep;
        MessageRateData.Add((_currentSimulationTime, messageRate));
    }

    public double GetAverageWorkerCount()
    {
        if (WorkerCountData.Count == 0) return 0;
        return WorkerCountData.Average(d => d.Value);
    }
}