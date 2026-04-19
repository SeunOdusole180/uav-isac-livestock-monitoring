#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/netanim-module.h"
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace ns3;

/**
 * Enhanced Gauss–Markov Mobility Model (EGM-style) in ONE FILE (Option A)
 * - Uses direction deviation (d_dev) rather than a global mean direction
 * - Adds smooth boundary avoidance by biasing mean deviation toward the region center
 * - Keeps temporally correlated motion via alpha
 *
 * Note: This is a practical ns-3 implementation aligned with the report’s EGM ideas,
 * while remaining easy to parametrize from your existing MobilityHelper setup.
 */
class EnhancedGaussMarkovMobilityModel : public MobilityModel
{
public:
  static TypeId GetTypeId (void)
  {
    static TypeId tid = TypeId ("ns3::EnhancedGaussMarkovMobilityModel")
      .SetParent<MobilityModel> ()
      .SetGroupName ("Mobility")
      .AddConstructor<EnhancedGaussMarkovMobilityModel> ()

      .AddAttribute ("Bounds",
                     "Bounds of the area in which the node moves.",
                     BoxValue (Box (-500, 500, -500, 500, 0, 200)),
                     MakeBoxAccessor (&EnhancedGaussMarkovMobilityModel::m_bounds),
                     MakeBoxChecker ())

      .AddAttribute ("TimeStep",
                     "Time step used to update the mobility state.",
                     TimeValue (Seconds (1.0)),
                     MakeTimeAccessor (&EnhancedGaussMarkovMobilityModel::m_timeStep),
                     MakeTimeChecker ())

      .AddAttribute ("Alpha",
                     "Gauss–Markov alpha (memory). 0 = memoryless, 1 = full memory.",
                     DoubleValue (0.85),
                     MakeDoubleAccessor (&EnhancedGaussMarkovMobilityModel::m_alpha),
                     MakeDoubleChecker<double> (0.0, 1.0))

      // Mean speed sampled once at initialization from this RV
      .AddAttribute ("MeanSpeed",
                     "Random variable for the long-term mean speed (sampled once at init).",
                     PointerValue (CreateObject<UniformRandomVariable> ()),
                     MakePointerAccessor (&EnhancedGaussMarkovMobilityModel::m_meanSpeedRv),
                     MakePointerChecker<RandomVariableStream> ())

      // Speed noise ~ N(0, sigma^2) inside the GM speed update
      .AddAttribute ("SpeedNoise",
                     "Normal RV for speed noise (Mean=0). Variance controls speed jitter.",
                     PointerValue (CreateObject<NormalRandomVariable> ()),
                     MakePointerAccessor (&EnhancedGaussMarkovMobilityModel::m_speedNoiseRv),
                     MakePointerChecker<RandomVariableStream> ())

      // Direction-deviation noise FAR from boundaries
      .AddAttribute ("DirDevNoiseFar",
                     "Normal RV for direction deviation noise when far from boundaries (Mean=0).",
                     PointerValue (CreateObject<NormalRandomVariable> ()),
                     MakePointerAccessor (&EnhancedGaussMarkovMobilityModel::m_dirDevNoiseFarRv),
                     MakePointerChecker<RandomVariableStream> ())

      // Direction-deviation noise NEAR boundaries (usually smaller variance for smoother turns)
      .AddAttribute ("DirDevNoiseNear",
                     "Normal RV for direction deviation noise when near boundaries (Mean=0).",
                     PointerValue (CreateObject<NormalRandomVariable> ()),
                     MakePointerAccessor (&EnhancedGaussMarkovMobilityModel::m_dirDevNoiseNearRv),
                     MakePointerChecker<RandomVariableStream> ())

      .AddAttribute ("Margin",
                     "Boundary margin. Within this distance from an edge, apply inward turning bias.",
                     DoubleValue (80.0),
                     MakeDoubleAccessor (&EnhancedGaussMarkovMobilityModel::m_margin),
                     MakeDoubleChecker<double> (0.0))

      .AddAttribute ("MaxTurnBiasDeg",
                     "Max magnitude (deg) of inward turning bias applied near boundaries (as mean d_dev target).",
                     DoubleValue (22.5),
                     MakeDoubleAccessor (&EnhancedGaussMarkovMobilityModel::m_maxTurnBiasDeg),
                     MakeDoubleChecker<double> (0.0, 180.0))

      .AddAttribute ("MaxDevStepDeg",
                     "Clamp for instantaneous direction deviation step (deg) to prevent sharp turns.",
                     DoubleValue (10.0),
                     MakeDoubleAccessor (&EnhancedGaussMarkovMobilityModel::m_maxDevStepDeg),
                     MakeDoubleChecker<double> (0.0, 180.0))

      .AddAttribute ("InitialDirection",
                     "Initial direction (radians). If negative, sampled uniformly in [0, 2*pi).",
                     DoubleValue (-1.0),
                     MakeDoubleAccessor (&EnhancedGaussMarkovMobilityModel::m_initialDirection),
                     MakeDoubleChecker<double> (-1.0, 1000.0));
    return tid;
  }

  EnhancedGaussMarkovMobilityModel ()
  {
    m_speed = 0.0;
    m_dir = 0.0;
    m_dirDev = 0.0;
    m_meanSpeed = 10.0;
    m_center = Vector (0.0, 0.0, 0.0);
    m_velocity = Vector (0.0, 0.0, 0.0);
    m_event = EventId ();
  }

  // Allow external control (e.g., future xApp logic) without rewriting attributes:
  void SetAlpha (double a) { m_alpha = std::clamp(a, 0.0, 1.0); }
  void SetMeanSpeedValue (double ms) { m_meanSpeed = std::max(0.0, ms); }

private:
  // --- MobilityModel interface ---
  virtual void DoInitialize (void) override
  {
    // Center of bounds for inward steering
    m_center = Vector ((m_bounds.xMin + m_bounds.xMax) / 2.0,
                       (m_bounds.yMin + m_bounds.yMax) / 2.0,
                       (m_bounds.zMin + m_bounds.zMax) / 2.0);

    // Sample mean speed once
    if (m_meanSpeedRv)
      {
        m_meanSpeed = std::max(0.0, m_meanSpeedRv->GetValue ());
      }

    // Initial direction
    if (m_initialDirection >= 0.0)
      {
        m_dir = m_initialDirection;
      }
    else
      {
        Ptr<UniformRandomVariable> u = CreateObject<UniformRandomVariable> ();
        m_dir = u->GetValue (0.0, 2.0 * M_PI);
      }

    // Start with a small deviation
    Ptr<UniformRandomVariable> udev = CreateObject<UniformRandomVariable> ();
    double initDevDeg = udev->GetValue (-5.0, 5.0);
    m_dirDev = DegreesToRadians (initDevDeg);

    // Start speed near mean
    m_speed = m_meanSpeed;

    // Schedule periodic updates
    m_event = Simulator::Schedule (m_timeStep, &EnhancedGaussMarkovMobilityModel::Update, this);

    MobilityModel::DoInitialize ();
  }

  virtual void DoDispose (void) override
  {
    if (m_event.IsPending ())
      {
        Simulator::Cancel (m_event);
      }
    MobilityModel::DoDispose ();
  }

  virtual Vector DoGetPosition (void) const override
  {
    return m_position;
  }

  virtual void DoSetPosition (const Vector &position) override
  {
    m_position = position;
    NotifyCourseChange ();
  }

  virtual Vector DoGetVelocity (void) const override
  {
    return m_velocity;
  }

  virtual int64_t DoAssignStreams (int64_t stream) override
  {
    // Best-effort stream assignment
    int64_t cur = stream;
    if (m_meanSpeedRv)        { m_meanSpeedRv->SetStream (cur++); }
    if (m_speedNoiseRv)       { m_speedNoiseRv->SetStream (cur++); }
    if (m_dirDevNoiseFarRv)   { m_dirDevNoiseFarRv->SetStream (cur++); }
    if (m_dirDevNoiseNearRv)  { m_dirDevNoiseNearRv->SetStream (cur++); }
    return (cur - stream);
  }
  virtual Ptr<MobilityModel> Copy () const override
{
  Ptr<EnhancedGaussMarkovMobilityModel> copy = CreateObject<EnhancedGaussMarkovMobilityModel> ();

  // Copy configuration/attributes
  copy->m_bounds = m_bounds;
  copy->m_timeStep = m_timeStep;
  copy->m_alpha = m_alpha;
  copy->m_margin = m_margin;
  copy->m_maxTurnBiasDeg = m_maxTurnBiasDeg;
  copy->m_maxDevStepDeg = m_maxDevStepDeg;
  copy->m_initialDirection = m_initialDirection;

  copy->m_meanSpeedRv = m_meanSpeedRv;
  copy->m_speedNoiseRv = m_speedNoiseRv;
  copy->m_dirDevNoiseFarRv = m_dirDevNoiseFarRv;
  copy->m_dirDevNoiseNearRv = m_dirDevNoiseNearRv;

  // Copy state (so it truly duplicates motion state)
  copy->m_position = m_position;
  copy->m_velocity = m_velocity;
  copy->m_center = m_center;

  copy->m_speed = m_speed;
  copy->m_dir = m_dir;
  copy->m_dirDev = m_dirDev;
  copy->m_meanSpeed = m_meanSpeed;

  return copy;
}


private:
  // --- Helpers ---
  static double WrapAngleRad (double a)
  {
    while (a > M_PI)  a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  }

  static double DegreesToRadians (double d)
  {
    return d * M_PI / 180.0;
  }

  bool IsNearBoundary (const Vector &p) const
  {
    bool nearLeft   = (p.x - m_bounds.xMin) < m_margin;
    bool nearRight  = (m_bounds.xMax - p.x) < m_margin;
    bool nearBottom = (p.y - m_bounds.yMin) < m_margin;
    bool nearTop    = (m_bounds.yMax - p.y) < m_margin;
    return nearLeft || nearRight || nearBottom || nearTop;
  }

  void ClampInsideBounds (Vector &p)
  {
    p.x = std::max (m_bounds.xMin, std::min (p.x, m_bounds.xMax));
    p.y = std::max (m_bounds.yMin, std::min (p.y, m_bounds.yMax));
    p.z = std::max (m_bounds.zMin, std::min (p.z, m_bounds.zMax));
  }

  void Update ()
  {
    const double dt = m_timeStep.GetSeconds ();
    if (dt <= 0.0)
      {
        m_event = Simulator::Schedule (m_timeStep, &EnhancedGaussMarkovMobilityModel::Update, this);
        return;
      }

    // === (1) Speed update (EGM/GM style) ===
    // s_t = a*s_{t-1} + (1-a)*s_bar + sqrt(1-a^2)*N(0, sigma_s^2)
    double noiseV = 0.0;
    if (m_speedNoiseRv) noiseV = m_speedNoiseRv->GetValue ();

    m_speed = m_alpha * m_speed
            + (1.0 - m_alpha) * m_meanSpeed
            + std::sqrt (std::max (0.0, 1.0 - m_alpha * m_alpha)) * noiseV;

    m_speed = std::max (0.0, m_speed); // no negative speed

    // === (2) Direction deviation update with smooth boundary avoidance ===
    bool near = IsNearBoundary (m_position);

    // Mean deviation target (radians). Far from boundary: 0 bias.
    double meanDev = 0.0;

    if (near)
      {
        // Steer inward toward center:
        double desired = std::atan2 (m_center.y - m_position.y, m_center.x - m_position.x);
        double err = WrapAngleRad (desired - m_dir);

        // Bias magnitude capped by MaxTurnBiasDeg (in radians)
        double maxBias = DegreesToRadians (m_maxTurnBiasDeg);
        meanDev = std::clamp (err, -maxBias, maxBias);
      }

    double noiseD = 0.0;
    if (near)
      {
        if (m_dirDevNoiseNearRv) noiseD = m_dirDevNoiseNearRv->GetValue ();
      }
    else
      {
        if (m_dirDevNoiseFarRv) noiseD = m_dirDevNoiseFarRv->GetValue ();
      }

    // d_dev_t = a*d_dev_{t-1} + (1-a)*meanDev + sqrt(1-a^2)*N(...)
    m_dirDev = m_alpha * m_dirDev
             + (1.0 - m_alpha) * meanDev
             + std::sqrt (std::max (0.0, 1.0 - m_alpha * m_alpha)) * noiseD;

    // Clamp instantaneous turn step
    double maxDev = DegreesToRadians (m_maxDevStepDeg);
    m_dirDev = std::clamp (m_dirDev, -maxDev, maxDev);

    // === (3) Direction update ===
    m_dir = WrapAngleRad (m_dir + m_dirDev);

    // === (4)(5) Position update ===
    Vector next = m_position;
    next.x += m_speed * dt * std::cos (m_dir);
    next.y += m_speed * dt * std::sin (m_dir);
    // keep altitude constant (your initial Z); just clamp to bounds
    next.z = m_position.z;

    // Clamp inside bounds (soft constraint)
    ClampInsideBounds (next);

    // Velocity estimate for logging/anim
    m_velocity = Vector ((next.x - m_position.x) / dt,
                         (next.y - m_position.y) / dt,
                         (next.z - m_position.z) / dt);

    m_position = next;
    NotifyCourseChange ();

    // Reschedule
    m_event = Simulator::Schedule (m_timeStep, &EnhancedGaussMarkovMobilityModel::Update, this);
  }

private:
  // Attributes / config
  Box m_bounds;
  Time m_timeStep;
  double m_alpha;
  double m_margin;
  double m_maxTurnBiasDeg;
  double m_maxDevStepDeg;
  double m_initialDirection;

  Ptr<RandomVariableStream> m_meanSpeedRv;
  Ptr<RandomVariableStream> m_speedNoiseRv;
  Ptr<RandomVariableStream> m_dirDevNoiseFarRv;
  Ptr<RandomVariableStream> m_dirDevNoiseNearRv;

  // State
  Vector m_position;
  Vector m_velocity;
  Vector m_center;

  double m_speed;     // current speed
  double m_dir;       // current direction (heading) radians
  double m_dirDev;    // current direction deviation radians
  double m_meanSpeed; // sampled once

  EventId m_event;
};

NS_OBJECT_ENSURE_REGISTERED (EnhancedGaussMarkovMobilityModel);


int main(int argc, char *argv[])
{
  Time::SetResolution(Time::NS);

  // Simulation parameters
  uint32_t numUAVs = 4;
  double simulationTime = 600.0;
  double areaSize = 800.0; // you comment 3km but set 800; leaving as-is

  // Create UAV nodes
  NodeContainer uavNodes;
  uavNodes.Create(numUAVs);

  // ==================== MOBILITY PATTERNS ====================
  // We keep your 4-role idea, but switch to ns3::EnhancedGaussMarkovMobilityModel
  // and set the EGM-style parameters:
  // - MeanSpeed sampled once
  // - SpeedNoise controls speed jitter
  // - DirDevNoiseFar/Near control heading jitter (near is smaller for smooth boundary turns)
  // - Margin + MaxTurnBiasDeg implements progressive boundary avoidance

  // Pattern 1: Surveillance UAV (Slow, High Persistence)
  MobilityHelper surveillanceUAV;
  surveillanceUAV.SetMobilityModel("ns3::EnhancedGaussMarkovMobilityModel",
      "Bounds", BoxValue(Box(-areaSize/2, areaSize/2, -areaSize/2, areaSize/2, 60, 80)),
      "TimeStep", TimeValue(Seconds(2.0)),
      "Alpha", DoubleValue(0.80),
      "Margin", DoubleValue(80.0),
      "MaxTurnBiasDeg", DoubleValue(22.5),
      "MaxDevStepDeg", DoubleValue(8.0),
      "MeanSpeed", PointerValue(CreateObjectWithAttributes<UniformRandomVariable>(
          "Min", DoubleValue(8.0), "Max", DoubleValue(15.0))),
      "SpeedNoise", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(2.0))),
      "DirDevNoiseFar", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(0.08))),
      "DirDevNoiseNear", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(0.02)))
  );

  // Pattern 2: Patrol UAV (Medium Speed, Balanced)
  MobilityHelper patrolUAV;
  patrolUAV.SetMobilityModel("ns3::EnhancedGaussMarkovMobilityModel",
      "Bounds", BoxValue(Box(-areaSize/2, areaSize/2, -areaSize/2, areaSize/2, 80, 95)),
      "TimeStep", TimeValue(Seconds(1.5)),
      "Alpha", DoubleValue(0.60),
      "Margin", DoubleValue(80.0),
      "MaxTurnBiasDeg", DoubleValue(22.5),
      "MaxDevStepDeg", DoubleValue(10.0),
      "MeanSpeed", PointerValue(CreateObjectWithAttributes<UniformRandomVariable>(
          "Min", DoubleValue(15.0), "Max", DoubleValue(25.0))),
      "SpeedNoise", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(3.0))),
      "DirDevNoiseFar", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(0.12))),
      "DirDevNoiseNear", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(0.03)))
  );

  // Pattern 3: Rapid Response UAV (Fast, More Agile)
  MobilityHelper rapidUAV;
  rapidUAV.SetMobilityModel("ns3::EnhancedGaussMarkovMobilityModel",
      "Bounds", BoxValue(Box(-areaSize/2, areaSize/2, -areaSize/2, areaSize/2, 95, 110)),
      "TimeStep", TimeValue(Seconds(1.0)),
      "Alpha", DoubleValue(0.35),
      "Margin", DoubleValue(80.0),
      "MaxTurnBiasDeg", DoubleValue(25.0),
      "MaxDevStepDeg", DoubleValue(12.0),
      "MeanSpeed", PointerValue(CreateObjectWithAttributes<UniformRandomVariable>(
          "Min", DoubleValue(20.0), "Max", DoubleValue(35.0))),
      "SpeedNoise", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(5.0))),
      "DirDevNoiseFar", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(0.20))),
      "DirDevNoiseNear", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(0.05)))
  );

  // Pattern 4: Strategic UAV (Stable, Wider sweep)
  MobilityHelper strategicUAV;
  strategicUAV.SetMobilityModel("ns3::EnhancedGaussMarkovMobilityModel",
      "Bounds", BoxValue(Box(-areaSize/2, areaSize/2, -areaSize/2, areaSize/2, 120, 130)),
      "TimeStep", TimeValue(Seconds(1.2)),
      "Alpha", DoubleValue(0.90),
      "Margin", DoubleValue(80.0),
      "MaxTurnBiasDeg", DoubleValue(20.0),
      "MaxDevStepDeg", DoubleValue(8.0),
      "MeanSpeed", PointerValue(CreateObjectWithAttributes<UniformRandomVariable>(
          "Min", DoubleValue(12.0), "Max", DoubleValue(22.0))),
      "SpeedNoise", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(4.0))),
      "DirDevNoiseFar", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(0.10))),
      "DirDevNoiseNear", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(0.02)))
  );

  // ==================== INITIAL POSITIONS ====================
  Ptr<ListPositionAllocator> initialPosition = CreateObject<ListPositionAllocator>();
  initialPosition->Add(Vector(-240, -240, 60));  // UAV 0 - SW corner
  initialPosition->Add(Vector(240, -240, 80));  // UAV 1 - SE corner
  initialPosition->Add(Vector(-240, 240, 100));   // UAV 2 - NW corner
  initialPosition->Add(Vector(240, 240, 120));    // UAV 3 - NE corner

  // Apply mobility patterns to UAVs (each helper installs 1 node)
  surveillanceUAV.SetPositionAllocator(initialPosition);
  surveillanceUAV.Install(uavNodes.Get(0));

  patrolUAV.SetPositionAllocator(initialPosition);
  patrolUAV.Install(uavNodes.Get(1));

  rapidUAV.SetPositionAllocator(initialPosition);
  rapidUAV.Install(uavNodes.Get(2));

  strategicUAV.SetPositionAllocator(initialPosition);
  strategicUAV.Install(uavNodes.Get(3));

  // ==================== NETANIM VISUALIZATION ====================
  AnimationInterface anim("uav-mobility.xml");

  anim.UpdateNodeColor(uavNodes.Get(0), 0, 255, 0);     // Green - Surveillance
  anim.UpdateNodeColor(uavNodes.Get(1), 0, 0, 255);     // Blue - Patrol
  anim.UpdateNodeColor(uavNodes.Get(2), 255, 0, 0);     // Red - Rapid Response
  anim.UpdateNodeColor(uavNodes.Get(3), 255, 165, 0);   // Orange - Strategic

  for (uint32_t i = 0; i < uavNodes.GetN(); ++i)
    {
      anim.UpdateNodeSize(i, 15, 15);
    }

  anim.SetMobilityPollInterval(Seconds(0.5));

  // ==================== CSV DATA COLLECTION ====================
  std::ofstream trajectoryFile("uav_trajectories.csv");
  trajectoryFile << "Time,UAV,Type,PositionX,PositionY,Altitude,VelocityX,VelocityY,VelocityZ,Speed" << std::endl;

  for (double time = 0; time <= simulationTime; time += 2.0)
    {
      Simulator::Schedule(Seconds(time), [time, uavNodes, &trajectoryFile]() {
        for (uint32_t i = 0; i < uavNodes.GetN(); ++i)
          {
            Ptr<MobilityModel> mobility = uavNodes.Get(i)->GetObject<MobilityModel>();
            Vector position = mobility->GetPosition();
            Vector velocity = mobility->GetVelocity();
            double speed = std::sqrt(velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z);

            const char* type = "";
            switch(i) {
              case 0: type = "Surveillance"; break;
              case 1: type = "Patrol"; break;
              case 2: type = "RapidResponse"; break;
              case 3: type = "Strategic"; break;
            }

            trajectoryFile << time << "," << i << "," << type << ","
                           << position.x << "," << position.y << "," << position.z << ","
                           << velocity.x << "," << velocity.y << "," << velocity.z << ","
                           << speed << std::endl;
          }
      });
    }

  // ==================== SIMULATION EXECUTION ====================
  Simulator::Stop(Seconds(simulationTime));
  Simulator::Run();
  Simulator::Destroy();

  trajectoryFile.close();

  std::cout << "UAV Mobility Simulation Completed!" << std::endl;
  std::cout << "Visual animation: uav-mobility.xml" << std::endl;
  std::cout << "Data for analysis: uav_trajectories.csv" << std::endl;

  return 0;
}
