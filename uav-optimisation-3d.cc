#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/nr-helper.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/nr-ue-net-device.h"
#include "ns3/nr-gnb-net-device.h"
#include <sys/stat.h>

#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cstring>
#include <sstream>
#include <string>

using namespace ns3;

// ==================== ENERGY HELPERS ====================
static inline double DbmToW(double pDbm)
{
  return std::pow(10.0, (pDbm - 30.0) / 10.0);
}

static inline double HoverPowerQuadrotor(double mKg,
                                         double rotorRadiusM,
                                         double rhoAir,
                                         double etaProp,
                                         uint32_t nRotors = 4)
{
  double g = 9.81;
  double Atot = nRotors * M_PI * rotorRadiusM * rotorRadiusM;
  double Pideal = std::pow(mKg * g, 1.5) / std::sqrt(2.0 * rhoAir * Atot);
  return Pideal / std::max(etaProp, 1e-9);
}

// ==================== 3D TERRAIN MODEL ====================
// Cows can now exist at varying elevations (hills, valleys)
class TerrainModel
{
public:
  TerrainModel(double xMin, double xMax, double yMin, double yMax,
               double baseZ = 0.0, double hillHeight = 20.0, double hillFreq = 0.01)
    : m_xMin(xMin), m_xMax(xMax), m_yMin(yMin), m_yMax(yMax),
      m_baseZ(baseZ), m_hillHeight(hillHeight), m_hillFreq(hillFreq)
  {}

  // Simple sinusoidal terrain height function
  double GetHeight(double x, double y) const
  {
    double zTerrain = m_baseZ +
      m_hillHeight * 0.5 * (std::sin(m_hillFreq * x * 2 * M_PI) +
                            std::sin(m_hillFreq * y * 2 * M_PI) +
                            0.5 * std::sin(m_hillFreq * (x + y) * M_PI));
    return std::max(0.0, zTerrain);
  }

private:
  double m_xMin, m_xMax, m_yMin, m_yMax;
  double m_baseZ;
  double m_hillHeight;
  double m_hillFreq;
};

/* ============================================================================
 *  EnhancedGaussMarkovMobilityModel CLASS (unchanged from 2D version)
 * ============================================================================
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

      .AddAttribute ("MeanSpeed",
                     "Random variable for the long-term mean speed (sampled once at init).",
                     PointerValue (CreateObject<UniformRandomVariable> ()),
                     MakePointerAccessor (&EnhancedGaussMarkovMobilityModel::m_meanSpeedRv),
                     MakePointerChecker<RandomVariableStream> ())

      .AddAttribute ("SpeedNoise",
                     "Normal RV for speed noise (Mean=0). Variance controls speed jitter.",
                     PointerValue (CreateObject<NormalRandomVariable> ()),
                     MakePointerAccessor (&EnhancedGaussMarkovMobilityModel::m_speedNoiseRv),
                     MakePointerChecker<RandomVariableStream> ())

      .AddAttribute ("DirDevNoiseFar",
                     "Normal RV for direction deviation noise when far from boundaries (Mean=0).",
                     PointerValue (CreateObject<NormalRandomVariable> ()),
                     MakePointerAccessor (&EnhancedGaussMarkovMobilityModel::m_dirDevNoiseFarRv),
                     MakePointerChecker<RandomVariableStream> ())

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
                     "Max magnitude (deg) of inward turning bias applied near boundaries.",
                     DoubleValue (22.5),
                     MakeDoubleAccessor (&EnhancedGaussMarkovMobilityModel::m_maxTurnBiasDeg),
                     MakeDoubleChecker<double> (0.0, 180.0))

      .AddAttribute ("MaxDevStepDeg",
                     "Clamp for instantaneous direction deviation step (deg).",
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

  void SetAlpha (double a) { m_alpha = std::clamp(a, 0.0, 1.0); }
  void SetMeanSpeedValue (double ms) { m_meanSpeed = std::max(0.0, ms); }

  void SetTargetAltitude (double z)
  {
    m_targetAltitude = std::max(m_bounds.zMin, std::min(z, m_bounds.zMax));
  }

  double GetTargetAltitude () const
  {
    return m_targetAltitude;
  }

private:
  virtual void DoInitialize (void) override
  {
    m_center = Vector ((m_bounds.xMin + m_bounds.xMax) / 2.0,
                       (m_bounds.yMin + m_bounds.yMax) / 2.0,
                       (m_bounds.zMin + m_bounds.zMax) / 2.0);

    if (m_meanSpeedRv)
      {
        m_meanSpeed = std::max(0.0, m_meanSpeedRv->GetValue ());
      }

    if (m_initialDirection >= 0.0)
      {
        m_dir = m_initialDirection;
      }
    else
      {
        Ptr<UniformRandomVariable> u = CreateObject<UniformRandomVariable> ();
        m_dir = u->GetValue (0.0, 2.0 * M_PI);
      }

    Ptr<UniformRandomVariable> udev = CreateObject<UniformRandomVariable> ();
    double initDevDeg = udev->GetValue (-5.0, 5.0);
    m_dirDev = DegreesToRadians (initDevDeg);

    m_speed = m_meanSpeed;

    m_targetAltitude = std::max(m_bounds.zMin, std::min(m_position.z, m_bounds.zMax));

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

    double noiseV = 0.0;
    if (m_speedNoiseRv) noiseV = m_speedNoiseRv->GetValue ();

    m_speed = m_alpha * m_speed
            + (1.0 - m_alpha) * m_meanSpeed
            + std::sqrt (std::max (0.0, 1.0 - m_alpha * m_alpha)) * noiseV;

    m_speed = std::max (0.0, m_speed);

    bool near = IsNearBoundary (m_position);

    double meanDev = 0.0;

    if (near)
      {
        double desired = std::atan2 (m_center.y - m_position.y, m_center.x - m_position.x);
        double err = WrapAngleRad (desired - m_dir);

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

    m_dirDev = m_alpha * m_dirDev
             + (1.0 - m_alpha) * meanDev
             + std::sqrt (std::max (0.0, 1.0 - m_alpha * m_alpha)) * noiseD;

    double maxDev = DegreesToRadians (m_maxDevStepDeg);
    m_dirDev = std::clamp (m_dirDev, -maxDev, maxDev);

    m_dir = WrapAngleRad (m_dir + m_dirDev);

    Vector next = m_position;
    next.x += m_speed * dt * std::cos (m_dir);
    next.y += m_speed * dt * std::sin (m_dir);

    double maxVz = 2.0;
    double zStep = maxVz * dt;
    double dz = m_targetAltitude - m_position.z;
    dz = std::max(-zStep, std::min(dz, zStep));
    next.z = m_position.z + dz;

    ClampInsideBounds (next);

    m_velocity = Vector ((next.x - m_position.x) / dt,
                         (next.y - m_position.y) / dt,
                         (next.z - m_position.z) / dt);

    m_position = next;
    NotifyCourseChange ();

    m_event = Simulator::Schedule (m_timeStep, &EnhancedGaussMarkovMobilityModel::Update, this);
  }

private:
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

  Vector m_position;
  Vector m_velocity;
  Vector m_center;

  double m_speed;
  double m_dir;
  double m_dirDev;
  double m_meanSpeed;
  double m_targetAltitude;

  EventId m_event;
};

NS_OBJECT_ENSURE_REGISTERED (EnhancedGaussMarkovMobilityModel);

// ============================================================================
// 3D ISAC Measurement Structure
// ============================================================================
struct IsacMeas3D
{
  bool detected{false};
  double snrDb{-1e9};
  double x{std::numeric_limits<double>::quiet_NaN()};
  double y{std::numeric_limits<double>::quiet_NaN()};
  double z{std::numeric_limits<double>::quiet_NaN()};  // NEW: z-coordinate
  uint32_t tagId{0};
  uint32_t uavId{0};  // Track which UAV made the detection
};

// ============================================================================
// 3D SNR-Weighted Fusion
// ============================================================================
static bool FuseSNRWeighted3D(const std::vector<IsacMeas3D>& meas,
                              double snrThDb,
                              double& xFused,
                              double& yFused,
                              double& zFused,
                              uint32_t& nUsed)
{
  nUsed = 0;
  double sumW = 0.0;
  double sumX = 0.0;
  double sumY = 0.0;
  double sumZ = 0.0;

  for (const auto& m : meas)
  {
    if (!m.detected) continue;
    if (!std::isfinite(m.x) || !std::isfinite(m.y) || !std::isfinite(m.z)) continue;
    if (m.snrDb < snrThDb) continue;

    double w = std::pow(10.0, m.snrDb / 10.0);
    if (!std::isfinite(w) || w <= 0) continue;

    sumW += w;
    sumX += w * m.x;
    sumY += w * m.y;
    sumZ += w * m.z;
    nUsed++;
  }

  if (nUsed == 0 || sumW <= 0.0)
  {
    xFused = std::numeric_limits<double>::quiet_NaN();
    yFused = std::numeric_limits<double>::quiet_NaN();
    zFused = std::numeric_limits<double>::quiet_NaN();
    return false;
  }

  xFused = sumX / sumW;
  yFused = sumY / sumW;
  zFused = sumZ / sumW;
  return true;
}

// ============================================================================
// 3D Cow Agent (now with terrain-aware z-coordinate)
// ============================================================================
struct CowAgent3D
{
  uint32_t id;
  Vector pos;     // (x, y, z) - z follows terrain
  Vector vel;     // (vx, vy, vz) - vz typically 0 for ground animals
  uint32_t tagId;
};

static double Clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

// ============================================================================
// Per-Role Statistics Structure
// ============================================================================
struct RoleStats
{
  uint32_t detections{0};
  uint32_t sensingAttempts{0};
  double sqErrSum{0.0};
  uint32_t rmseCount{0};
  double energyJ{0.0};
  double sensingTimeS{0.0};
  double commTimeS{0.0};
  uint64_t dataVolumeBytes{0};
  std::unordered_set<uint32_t> uniqueCowsDetected;
  double coveredAreaM2{0.0};
  
  // Grid for area coverage tracking
  std::unordered_set<uint64_t> coveredCells;
};

// ============================================================================
// 3D ISAC Sensing Application
// ============================================================================
class IsacSensingApp3D : public Application
{
public:
  IsacSensingApp3D() = default;

  void Configure(Ptr<Node> uav,
                 uint32_t uavId,
                 uint32_t roleId,  // NEW: 0=surv, 1=pat, 2=rapid, 3=strat
                 std::vector<CowAgent3D>* cows,
                 TerrainModel* terrain,
                 Ipv4Address remoteAddr,
                 uint16_t remotePort,
                 Time sensingInterval,
                 double snrThresholdDb,
                 double sigmaPosMeters,
                 double fcHz,
                 double ptDbm,
                 double gDb,
                 double rcsDbsm,
                 double noiseDbm,
                 double areaSize)
  {
    m_uav = uav;
    m_uavId = uavId;
    m_roleId = roleId;
    m_cows = cows;
    m_terrain = terrain;
    m_remoteAddr = remoteAddr;
    m_remotePort = remotePort;
    m_sensingInterval = sensingInterval;

    m_snrThDb = snrThresholdDb;
    m_sigmaPos = sigmaPosMeters;

    m_fc = fcHz;
    m_ptDbm = ptDbm;
    m_gDb = gDb;
    m_rcsDbsm = rcsDbsm;
    m_noiseDbm = noiseDbm;

    m_areaSize = areaSize;
    m_cellSize = 20.0;  // 20m grid cells for coverage tracking

    m_c = 3e8;
    m_lambda = m_c / m_fc;

    m_ptW = std::pow(10.0, (m_ptDbm - 30.0) / 10.0);
    m_gLin = std::pow(10.0, (m_gDb) / 10.0);
    m_rcsLin = std::pow(10.0, (m_rcsDbsm) / 10.0);
    m_noiseW = std::pow(10.0, (m_noiseDbm - 30.0) / 10.0);

    m_rng = CreateObject<NormalRandomVariable>();
    m_rng->SetAttribute("Mean", DoubleValue(0.0));
    m_rng->SetAttribute("Variance", DoubleValue(m_sigmaPos * m_sigmaPos));

    m_rngZ = CreateObject<NormalRandomVariable>();
    m_rngZ->SetAttribute("Mean", DoubleValue(0.0));
    m_rngZ->SetAttribute("Variance", DoubleValue(m_sigmaPos * m_sigmaPos * 0.5));  // Less z-noise
  }

  void ConfigureProcessingEnergy(double kappa, double c0, double c1, double fProcHz)
  {
    m_kappa = kappa;
    m_c0 = c0;
    m_c1 = c1;
    m_fProcHz = fProcHz;
  }

  double GetProcessingEnergyJ() const { return m_eProcJ; }
  uint32_t GetRoleId() const { return m_roleId; }
  
  RoleStats& GetRoleStats() { return m_stats; }
  const RoleStats& GetRoleStats() const { return m_stats; }

  double GetSensingIntervalSeconds() const { return m_sensingInterval.GetSeconds(); }

  void SetLogFiles(std::ofstream* reportCsv) { m_reportCsv = reportCsv; }

private:
  void StartApplication() override
  {
    m_socket = Socket::CreateSocket(m_uav, UdpSocketFactory::GetTypeId());
    m_socket->Connect(InetSocketAddress(m_remoteAddr, m_remotePort));

    m_running = true;
    ScheduleNext();
  }

  void StopApplication() override
  {
    m_running = false;
    if (m_sendEvent.IsPending()) Simulator::Cancel(m_sendEvent);
    if (m_socket) m_socket->Close();
  }

  void ScheduleNext()
  {
    if (!m_running) return;
    m_sendEvent = Simulator::Schedule(m_sensingInterval, &IsacSensingApp3D::SenseAndSend, this);
  }

  // 3D distance calculation for radar equation
  double ComputeSnrDb3D(const Vector& uavPos, const Vector& cowPos) const
  {
    double dx = uavPos.x - cowPos.x;
    double dy = uavPos.y - cowPos.y;
    double dz = uavPos.z - cowPos.z;
    double R = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    double denom = (std::pow(4.0 * M_PI, 3.0) * std::pow(R, 4.0)) + 1e-30;
    double pr = (m_ptW * (m_gLin * m_gLin) * (m_lambda * m_lambda) * m_rcsLin) / denom;
    double snrLin = pr / std::max(m_noiseW, 1e-30);
    return 10.0 * std::log10(std::max(snrLin, 1e-12));
  }

  // Track area coverage using grid cells
  void UpdateCoverage(const Vector& uavPos)
  {
    // Calculate sensing footprint based on altitude
    double footprintRadius = std::max(10.0, uavPos.z * 0.5);  // Rough approximation
    
    int cellsX = static_cast<int>(m_areaSize / m_cellSize);
    int cellsY = static_cast<int>(m_areaSize / m_cellSize);
    
    double halfArea = m_areaSize / 2.0;
    
    for (double dx = -footprintRadius; dx <= footprintRadius; dx += m_cellSize)
    {
      for (double dy = -footprintRadius; dy <= footprintRadius; dy += m_cellSize)
      {
        if (dx*dx + dy*dy <= footprintRadius*footprintRadius)
        {
          double cx = uavPos.x + dx;
          double cy = uavPos.y + dy;
          
          if (cx >= -halfArea && cx < halfArea && cy >= -halfArea && cy < halfArea)
          {
            int ix = static_cast<int>((cx + halfArea) / m_cellSize);
            int iy = static_cast<int>((cy + halfArea) / m_cellSize);
            uint64_t cellId = static_cast<uint64_t>(iy) * cellsX + ix;
            m_stats.coveredCells.insert(cellId);
          }
        }
      }
    }
  }

  void SenseAndSend()
  {
    Ptr<MobilityModel> mob = m_uav->GetObject<MobilityModel>();
    Vector u = mob->GetPosition();

    // Track sensing time
    double sensingDuration = m_sensingInterval.GetSeconds();
    m_stats.sensingTimeS += sensingDuration;
    m_stats.sensingAttempts++;

    // Update coverage
    UpdateCoverage(u);

    std::vector<uint8_t> buf;
    auto push_u16 = [&](uint16_t v) {
      buf.push_back(uint8_t(v & 0xFF));
      buf.push_back(uint8_t((v >> 8) & 0xFF));
    };
    auto push_f32 = [&](float f) {
      uint32_t x;
      std::memcpy(&x, &f, 4);
      buf.push_back(uint8_t(x & 0xFF));
      buf.push_back(uint8_t((x >> 8) & 0xFF));
      buf.push_back(uint8_t((x >> 16) & 0xFF));
      buf.push_back(uint8_t((x >> 24) & 0xFF));
    };

    uint16_t numDet = 0;
    push_u16((uint16_t)m_uavId);
    push_u16((uint16_t)m_roleId);  // Include role in packet
    push_u16(0);  // placeholder for numDet

    double t = Simulator::Now().GetSeconds();

    for (auto& cow : *m_cows)
    {
      double snrDb = ComputeSnrDb3D(u, cow.pos);
      bool detected = (snrDb >= m_snrThDb);

      // Stochastic detection near threshold
      if (detected)
      {
        double margin = snrDb - m_snrThDb;
        if (margin < 6.0)
        {
          double p = Clamp(margin / 6.0, 0.0, 1.0);
          Ptr<UniformRandomVariable> urv = CreateObject<UniformRandomVariable>();
          if (urv->GetValue(0.0, 1.0) > p)
          {
            detected = false;
          }
        }
      }

      if (!detected)
      {
        if (m_reportCsv)
        {
          (*m_reportCsv) << t << "," << m_uavId << "," << m_roleId << "," 
                         << cow.id << "," << cow.tagId
                         << ",0," << snrDb << ",nan,nan,nan\n";
        }
        continue;
      }

      // 3D position estimate with noise
      double xhat = cow.pos.x + m_rng->GetValue();
      double yhat = cow.pos.y + m_rng->GetValue();
      double zhat = cow.pos.z + m_rngZ->GetValue();

      push_u16((uint16_t)cow.id);
      push_u16((uint16_t)cow.tagId);
      push_f32((float)xhat);
      push_f32((float)yhat);
      push_f32((float)zhat);  // NEW: include z
      push_f32((float)snrDb);

      numDet++;
      m_stats.detections++;
      m_stats.uniqueCowsDetected.insert(cow.id);

      if (m_reportCsv)
      {
        (*m_reportCsv) << t << "," << m_uavId << "," << m_roleId << ","
                       << cow.id << "," << cow.tagId
                       << ",1," << snrDb << "," << xhat << "," << yhat << "," << zhat << "\n";
      }
    }

    // Patch numDet
    buf[4] = uint8_t(numDet & 0xFF);
    buf[5] = uint8_t((numDet >> 8) & 0xFF);

    Ptr<Packet> p = Create<Packet>(buf.data(), buf.size());
    m_socket->Send(p);

    m_stats.dataVolumeBytes += buf.size();

    // Processing energy
    double eProcStepJ = m_kappa * (m_c0 + m_c1 * static_cast<double>(numDet)) * m_fProcHz * m_fProcHz;
    m_eProcJ += eProcStepJ;

    ScheduleNext();
  }

private:
  Ptr<Node> m_uav;
  uint32_t m_uavId = 0;
  uint32_t m_roleId = 0;
  std::vector<CowAgent3D>* m_cows = nullptr;
  TerrainModel* m_terrain = nullptr;

  Ptr<Socket> m_socket;
  bool m_running = false;
  EventId m_sendEvent;

  Ipv4Address m_remoteAddr;
  uint16_t m_remotePort = 9000;
  Time m_sensingInterval = Seconds(1.0);

  double m_snrThDb = 30.0;
  double m_sigmaPos = 5.0;

  double m_fc = 3.5e9;
  double m_c = 3e8;
  double m_lambda = 0.0857;

  double m_ptDbm = 30.0;
  double m_gDb = 20.0;
  double m_rcsDbsm = -5.0;
  double m_noiseDbm = -110.0;

  double m_ptW = 1.0;
  double m_gLin = 100.0;
  double m_rcsLin = 0.316;
  double m_noiseW = 1e-14;

  double m_areaSize = 800.0;
  double m_cellSize = 20.0;

  Ptr<NormalRandomVariable> m_rng;
  Ptr<NormalRandomVariable> m_rngZ;

  std::ofstream* m_reportCsv = nullptr;

  double m_kappa = 1e-27;
  double m_c0 = 1e7;
  double m_c1 = 5e6;
  double m_fProcHz = 1e9;

  double m_eProcJ = 0.0;

  RoleStats m_stats;
};

// ============================================================================
// 3D Fusion Receiver Application
// ============================================================================
class IsacFusionReceiverApp3D : public Application
{
public:
  void Configure(uint16_t listenPort,
                 uint32_t numUavs,
                 uint32_t numCows,
                 double snrThDb,
                 std::ofstream* fusedCsv)
  {
    m_port = listenPort;
    m_numUavs = numUavs;
    m_numCows = numCows;
    m_snrThDb = snrThDb;
    m_fusedCsv = fusedCsv;

    m_meas.resize(m_numCows);
    for (auto& v : m_meas)
    {
      v.resize(m_numUavs);
    }

    m_roleStats.resize(4);  // 4 roles

    m_windowDetected = 0;
    m_windowTotal = 0;
    m_windowSqErrSum = 0.0;
    m_windowRmseCount = 0;
  }

  void SetCowTruthSource(std::vector<CowAgent3D>* cows) { m_cows = cows; }

  double GetWindowPdet() const
  {
    return (m_windowTotal > 0) ? (double(m_windowDetected) / double(m_windowTotal)) : 0.0;
  }

  double GetWindowRmseM() const
  {
    return (m_windowRmseCount > 0) ? std::sqrt(m_windowSqErrSum / double(m_windowRmseCount)) : 999.0;
  }

  double GetRunPdet() const
  {
    return (m_runTotal > 0) ? (double(m_runDetected) / double(m_runTotal)) : 0.0;
  }

  double GetRunRmseM() const
  {
    return (m_runRmseCount > 0) ? std::sqrt(m_runSqErrSum / double(m_runRmseCount)) : 999.0;
  }

  // Per-role accessors
  double GetRolePdet(uint32_t roleId) const
  {
    if (roleId >= m_roleStats.size()) return 0.0;
    const auto& rs = m_roleStats[roleId];
    return (rs.total > 0) ? (double(rs.detected) / double(rs.total)) : 0.0;
  }

  double GetRoleRmseM(uint32_t roleId) const
  {
    if (roleId >= m_roleStats.size()) return 999.0;
    const auto& rs = m_roleStats[roleId];
    return (rs.rmseCount > 0) ? std::sqrt(rs.sqErrSum / double(rs.rmseCount)) : 999.0;
  }

  uint32_t GetRoleDetections(uint32_t roleId) const
  {
    if (roleId >= m_roleStats.size()) return 0;
    return m_roleStats[roleId].detected;
  }

  void ResetWindowStats()
  {
    m_windowDetected = 0;
    m_windowTotal = 0;
    m_windowSqErrSum = 0.0;
    m_windowRmseCount = 0;
  }

private:
  struct RoleFusionStats
  {
    uint32_t detected{0};
    uint32_t total{0};
    double sqErrSum{0.0};
    uint32_t rmseCount{0};
  };

  void StartApplication() override
  {
    m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
    InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
    m_socket->Bind(local);
    m_socket->SetRecvCallback(MakeCallback(&IsacFusionReceiverApp3D::HandleRead, this));
  }

  void StopApplication() override
  {
    if (m_socket) m_socket->Close();
  }

  static uint16_t ReadU16(const uint8_t* p)
  {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
  }

  static float ReadF32(const uint8_t* p)
  {
    float f;
    uint32_t x = uint32_t(p[0]) |
                 (uint32_t(p[1]) << 8) |
                 (uint32_t(p[2]) << 16) |
                 (uint32_t(p[3]) << 24);
    std::memcpy(&f, &x, 4);
    return f;
  }

  void HandleRead(Ptr<Socket> socket)
  {
    Address from;
    Ptr<Packet> packet;

    while ((packet = socket->RecvFrom(from)))
    {
      uint32_t size = packet->GetSize();
      std::vector<uint8_t> data(size);
      packet->CopyData(data.data(), size);

      if (size < 6) return;

      uint16_t uavId = ReadU16(&data[0]);
      uint16_t roleId = ReadU16(&data[2]);
      uint16_t numDet = ReadU16(&data[4]);

      uint32_t offset = 6;
      double t = Simulator::Now().GetSeconds();

      for (uint16_t i = 0; i < numDet; ++i)
      {
        if (offset + 2 + 2 + 4 + 4 + 4 + 4 > size) break;

        uint16_t cowId = ReadU16(&data[offset]); offset += 2;
        uint16_t tagId = ReadU16(&data[offset]); offset += 2;
        float xhat = ReadF32(&data[offset]); offset += 4;
        float yhat = ReadF32(&data[offset]); offset += 4;
        float zhat = ReadF32(&data[offset]); offset += 4;
        float snr  = ReadF32(&data[offset]); offset += 4;

        if (cowId >= m_numCows || uavId >= m_numUavs) continue;

        IsacMeas3D m;
        m.detected = true;
        m.snrDb = snr;
        m.x = xhat;
        m.y = yhat;
        m.z = zhat;
        m.tagId = tagId;
        m.uavId = uavId;

        m_meas[cowId][uavId] = m;
      }

      // Fuse measurements for each cow
      for (uint32_t cowId = 0; cowId < m_numCows; ++cowId)
      {
        std::vector<IsacMeas3D> vec;
        vec.reserve(m_numUavs);
        for (uint32_t u = 0; u < m_numUavs; ++u) vec.push_back(m_meas[cowId][u]);

        double xf, yf, zf;
        uint32_t nUsed;
        bool detF = FuseSNRWeighted3D(vec, m_snrThDb, xf, yf, zf, nUsed);

        uint32_t tagId = (vec.size() > 0 ? vec[0].tagId : (1000 + cowId));

        if (m_fusedCsv)
        {
          (*m_fusedCsv) << t << "," << cowId << "," << tagId
                        << "," << (detF ? 1 : 0)
                        << "," << nUsed
                        << "," << xf << "," << yf << "," << zf << "\n";
        }

        m_windowTotal++;
        m_runTotal++;

        // Track per-role stats based on which UAV contributed most
        uint32_t primaryRole = roleId;  // Use the role from current packet

        if (primaryRole < m_roleStats.size())
        {
          m_roleStats[primaryRole].total++;
        }

        if (detF)
        {
          m_windowDetected++;
          m_runDetected++;

          if (primaryRole < m_roleStats.size())
          {
            m_roleStats[primaryRole].detected++;
          }

          if (m_cows && cowId < m_cows->size())
          {
            double dx = xf - (*m_cows)[cowId].pos.x;
            double dy = yf - (*m_cows)[cowId].pos.y;
            double dz = zf - (*m_cows)[cowId].pos.z;
            double se3d = dx * dx + dy * dy + dz * dz;

            m_windowSqErrSum += se3d;
            m_windowRmseCount++;

            m_runSqErrSum += se3d;
            m_runRmseCount++;

            if (primaryRole < m_roleStats.size())
            {
              m_roleStats[primaryRole].sqErrSum += se3d;
              m_roleStats[primaryRole].rmseCount++;
            }
          }
        }
      }
    }
  }

private:
  Ptr<Socket> m_socket;
  uint16_t m_port{9000};
  uint32_t m_numUavs{4};
  uint32_t m_numCows{20};
  double m_snrThDb{30.0};

  std::ofstream* m_fusedCsv{nullptr};
  std::vector<CowAgent3D>* m_cows{nullptr};

  std::vector<std::vector<IsacMeas3D>> m_meas;

  uint32_t m_windowDetected{0};
  uint32_t m_windowTotal{0};
  double m_windowSqErrSum{0.0};
  uint32_t m_windowRmseCount{0};

  uint32_t m_runDetected{0};
  uint32_t m_runTotal{0};
  double m_runSqErrSum{0.0};
  uint32_t m_runRmseCount{0};

  std::vector<RoleFusionStats> m_roleStats;
};

// ============================================================================
// Helper Functions
// ============================================================================
bool FileEmptyOrMissing(const std::string& name)
{
  struct stat st;
  if (stat(name.c_str(), &st) != 0) return true;
  return (st.st_size == 0);
}

static double ClampToRange(double v, double lo, double hi)
{
  return std::max(lo, std::min(v, hi));
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[])
{
  Time::SetResolution(Time::NS);

  // --- Sim parameters ---
  uint32_t numUAVs = 4;
  double simulationTime = 300.0;
  double areaSize = 800.0;
  uint16_t port = 9000;
  double bwMHz = 20.0;

  double telemetryRateMbps = 5.0;
  uint16_t telemetryPort = 9100;
  uint32_t telemetryPktSize = 1200;
  double telemetryInterval = 0.1;

  // gNB altitude
  double gnbZ = 50.0;

  // Terrain parameters
  double terrainBaseZ = 0.0;
  double terrainHillHeight = 20.0;
  double terrainHillFreq = 0.01;

  // Altitude ranges for each UAV role
  double survZmin = 60.0,  survZmax = 80.0;
  double patZmin  = 80.0,  patZmax  = 95.0;
  double rapidZmin= 95.0,  rapidZmax= 110.0;
  double stratZmin= 120.0, stratZmax= 130.0;

  // Mobility timesteps
  double survDt = 2.0;
  double patDt  = 1.5;
  double rapidDt= 1.0;
  double stratDt= 1.2;

  // GM memory (alpha)
  double alphaSurv  = 0.80;
  double alphaPat   = 0.60;
  double alphaRapid = 0.35;
  double alphaStrat = 0.90;

  // Speeds
  double survVmin = 8.0,  survVmax = 15.0;
  double patVmin  = 15.0, patVmax  = 25.0;
  double rapidVmin= 20.0, rapidVmax= 35.0;
  double stratVmin= 12.0, stratVmax= 22.0;

  double margin = 80.0;

  double survMaxTurnBiasDeg = 22.5;
  double patMaxTurnBiasDeg  = 22.5;
  double rapidMaxTurnBiasDeg= 25.0;
  double stratMaxTurnBiasDeg= 20.0;

  double survMaxDevStepDeg = 8.0;
  double patMaxDevStepDeg  = 10.0;
  double rapidMaxDevStepDeg= 12.0;
  double stratMaxDevStepDeg= 8.0;

  double survSpeedVar = 2.0;
  double patSpeedVar  = 3.0;
  double rapidSpeedVar= 5.0;
  double stratSpeedVar= 4.0;

  double survDirFarVar = 0.08;
  double patDirFarVar  = 0.12;
  double rapidDirFarVar= 0.20;
  double stratDirFarVar= 0.10;

  double survDirNearVar = 0.02;
  double patDirNearVar  = 0.03;
  double rapidDirNearVar= 0.05;
  double stratDirNearVar= 0.02;

  double uav0z = 60, uav1z = 80, uav2z = 100, uav3z = 120;

  // ISAC params
  uint32_t numCows = 80;
  double cowSpeedMax = 1.0;
  double cowUpdateDt = 1.0;
  double survSenseDt  = 1.0;
  double patSenseDt   = 1.0;
  double rapidSenseDt = 1.0;
  double stratSenseDt = 1.0;
  double snrThresholdDb = 30.0;
  double sigmaPosMeters = 5.0;

  double fc = 3.5e9;
  double ptDbm = 30.0;
  double gDb = 20.0;
  double rcsDbsm = -5.0;
  double noiseDbm = -110.0;

  double ueTxPowerDbm = 23.0;
  double gnbTxPowerDbm = 38.0;

  // Energy model params
  double uavMassKg = 2.8;
  double rotorRadiusM = 0.15;
  double rhoAir = 1.225;
  double etaProp = 0.7;
  double cv = 0.1;
  double energySampleDt = 1.0;

  double kappaProc = 1e-27;
  double c0Proc = 1e7;
  double c1Proc = 5e6;
  double fProcHz = 1e9;

  CommandLine cmd(__FILE__);
  uint32_t rngRun = 1;

  cmd.AddValue("RngRun", "Seed run number", rngRun);
  cmd.AddValue("simulationTime", "Simulation time (s)", simulationTime);
  cmd.AddValue("ueTxPower", "UE Tx power in dBm", ueTxPowerDbm);
  cmd.AddValue("gnbTxPower", "gNB Tx power in dBm", gnbTxPowerDbm);

  cmd.AddValue("numCows", "Number of cow agents", numCows);
  cmd.AddValue("snrTh", "ISAC detection threshold (dB)", snrThresholdDb);
  cmd.AddValue("survSenseDt",  "Surveillance sensing interval (s)", survSenseDt);
  cmd.AddValue("patSenseDt",   "Patrol sensing interval (s)", patSenseDt);
  cmd.AddValue("rapidSenseDt", "Rapid sensing interval (s)", rapidSenseDt);
  cmd.AddValue("stratSenseDt", "Strategic sensing interval (s)", stratSenseDt);
  cmd.AddValue("sigmaPos", "ISAC position noise std dev (m)", sigmaPosMeters);

  //cmd.AddValue("terrainHillHeight", "Terrain hill height (m)", terrainHillHeight);
  //cmd.AddValue("terrainHillFreq", "Terrain hill frequency", terrainHillFreq);

  cmd.AddValue("alphaSurv",  "Alpha for Surveillance UAV", alphaSurv);
  cmd.AddValue("alphaPat",   "Alpha for Patrol UAV", alphaPat);
  cmd.AddValue("alphaRapid", "Alpha for Rapid UAV", alphaRapid);
  cmd.AddValue("alphaStrat", "Alpha for Strategic UAV", alphaStrat);

  cmd.AddValue("survVmin", "Surveillance min speed (m/s)", survVmin);
  cmd.AddValue("survVmax", "Surveillance max speed (m/s)", survVmax);
  cmd.AddValue("patVmin",  "Patrol min speed (m/s)", patVmin);
  cmd.AddValue("patVmax",  "Patrol max speed (m/s)", patVmax);
  cmd.AddValue("rapidVmin","Rapid min speed (m/s)", rapidVmin);
  cmd.AddValue("rapidVmax","Rapid max speed (m/s)", rapidVmax);
  cmd.AddValue("stratVmin","Strategic min speed (m/s)", stratVmin);
  cmd.AddValue("stratVmax","Strategic max speed (m/s)", stratVmax);

  cmd.AddValue("survZmin","Surveillance zMin",survZmin);
  cmd.AddValue("survZmax","Surveillance zMax",survZmax);
  cmd.AddValue("patZmin","Patrol zMin",patZmin);
  cmd.AddValue("patZmax","Patrol zMax",patZmax);
  cmd.AddValue("rapidZmin","Rapid zMin",rapidZmin);
  cmd.AddValue("rapidZmax","Rapid zMax",rapidZmax);
  cmd.AddValue("stratZmin","Strategic zMin",stratZmin);
  cmd.AddValue("stratZmax","Strategic zMax",stratZmax);

  cmd.AddValue("survDt","Surveillance timestep (s)",survDt);
  cmd.AddValue("patDt","Patrol timestep (s)",patDt);
  cmd.AddValue("rapidDt","Rapid timestep (s)",rapidDt);
  cmd.AddValue("stratDt","Strategic timestep (s)",stratDt);

  cmd.AddValue("telemetryPktSize", "Telemetry packet size (bytes)", telemetryPktSize);
  cmd.AddValue("telemetryInterval", "Telemetry packet interval (s)", telemetryInterval);
  cmd.AddValue("telemetryPort", "Telemetry UDP port", telemetryPort);
  cmd.AddValue("bwMHz", "NR bandwidth in MHz", bwMHz);
  cmd.AddValue("telemetryRateMbps", "Background telemetry rate per UAV (Mbps)", telemetryRateMbps);

  cmd.AddValue("uavMassKg", "UAV mass (kg)", uavMassKg);
  cmd.AddValue("rotorRadiusM", "Rotor radius (m)", rotorRadiusM);
  cmd.AddValue("rhoAir", "Air density (kg/m^3)", rhoAir);
  cmd.AddValue("etaProp", "Propulsion efficiency", etaProp);
  cmd.AddValue("cv", "Velocity-squared propulsion coefficient", cv);
  cmd.AddValue("energySampleDt", "Energy sampling period (s)", energySampleDt);

  cmd.AddValue("kappaProc", "Processing energy coefficient", kappaProc);
  cmd.AddValue("c0Proc", "Baseline CPU cycles per sensing step", c0Proc);
  cmd.AddValue("c1Proc", "Extra CPU cycles per detected target", c1Proc);
  cmd.AddValue("fProcHz", "Processor frequency (Hz)", fProcHz);

  cmd.Parse(argc, argv);
  RngSeedManager::SetSeed(1);
  RngSeedManager::SetRun(rngRun);

  // Create terrain model
  TerrainModel terrain(-areaSize/2, areaSize/2, -areaSize/2, areaSize/2,
                       terrainBaseZ, terrainHillHeight, terrainHillFreq);

  // --- Nodes ---
  NodeContainer uavNodes;
  uavNodes.Create(numUAVs);

  NodeContainer gnbNodes;
  gnbNodes.Create(1);

  // gNB mobility
  MobilityHelper gnbMobility;
  gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  gnbMobility.Install(gnbNodes);
  gnbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0.0, 0.0, gnbZ));

  // UAV mobility patterns (same as 2D, using your EnhancedGaussMarkovMobilityModel)
  Ptr<ListPositionAllocator> initialPosition = CreateObject<ListPositionAllocator>();
  initialPosition->Add(Vector(-240, -240, uav0z));
  initialPosition->Add(Vector(240, -240, uav1z));
  initialPosition->Add(Vector(-240, 240, uav2z));
  initialPosition->Add(Vector(240, 240, uav3z));

  // Pattern 1: Surveillance
  MobilityHelper surveillanceUAV;
  surveillanceUAV.SetMobilityModel("ns3::EnhancedGaussMarkovMobilityModel",
      "Bounds", BoxValue(Box(-areaSize/2, areaSize/2, -areaSize/2, areaSize/2, survZmin, survZmax)),
      "TimeStep", TimeValue(Seconds(survDt)),
      "Alpha", DoubleValue(alphaSurv),
      "Margin", DoubleValue(margin),
      "MaxTurnBiasDeg", DoubleValue(survMaxTurnBiasDeg),
      "MaxDevStepDeg", DoubleValue(survMaxDevStepDeg),
      "MeanSpeed", PointerValue(CreateObjectWithAttributes<UniformRandomVariable>(
          "Min", DoubleValue(survVmin), "Max", DoubleValue(survVmax))),
      "SpeedNoise", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(survSpeedVar))),
      "DirDevNoiseFar", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(survDirFarVar))),
      "DirDevNoiseNear", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(survDirNearVar)))
  );

  // Pattern 2: Patrol
  MobilityHelper patrolUAV;
  patrolUAV.SetMobilityModel("ns3::EnhancedGaussMarkovMobilityModel",
      "Bounds", BoxValue(Box(-areaSize/2, areaSize/2, -areaSize/2, areaSize/2, patZmin, patZmax)),
      "TimeStep", TimeValue(Seconds(patDt)),
      "Alpha", DoubleValue(alphaPat),
      "Margin", DoubleValue(margin),
      "MaxTurnBiasDeg", DoubleValue(patMaxTurnBiasDeg),
      "MaxDevStepDeg", DoubleValue(patMaxDevStepDeg),
      "MeanSpeed", PointerValue(CreateObjectWithAttributes<UniformRandomVariable>(
          "Min", DoubleValue(patVmin), "Max", DoubleValue(patVmax))),
      "SpeedNoise", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(patSpeedVar))),
      "DirDevNoiseFar", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(patDirFarVar))),
      "DirDevNoiseNear", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(patDirNearVar)))
  );

  // Pattern 3: Rapid
  MobilityHelper rapidUAV;
  rapidUAV.SetMobilityModel("ns3::EnhancedGaussMarkovMobilityModel",
      "Bounds", BoxValue(Box(-areaSize/2, areaSize/2, -areaSize/2, areaSize/2, rapidZmin, rapidZmax)),
      "TimeStep", TimeValue(Seconds(rapidDt)),
      "Alpha", DoubleValue(alphaRapid),
      "Margin", DoubleValue(margin),
      "MaxTurnBiasDeg", DoubleValue(rapidMaxTurnBiasDeg),
      "MaxDevStepDeg", DoubleValue(rapidMaxDevStepDeg),
      "MeanSpeed", PointerValue(CreateObjectWithAttributes<UniformRandomVariable>(
          "Min", DoubleValue(rapidVmin), "Max", DoubleValue(rapidVmax))),
      "SpeedNoise", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(rapidSpeedVar))),
      "DirDevNoiseFar", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(rapidDirFarVar))),
      "DirDevNoiseNear", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(rapidDirNearVar)))
  );

  // Pattern 4: Strategic
  MobilityHelper strategicUAV;
  strategicUAV.SetMobilityModel("ns3::EnhancedGaussMarkovMobilityModel",
      "Bounds", BoxValue(Box(-areaSize/2, areaSize/2, -areaSize/2, areaSize/2, stratZmin, stratZmax)),
      "TimeStep", TimeValue(Seconds(stratDt)),
      "Alpha", DoubleValue(alphaStrat),
      "Margin", DoubleValue(margin),
      "MaxTurnBiasDeg", DoubleValue(stratMaxTurnBiasDeg),
      "MaxDevStepDeg", DoubleValue(stratMaxDevStepDeg),
      "MeanSpeed", PointerValue(CreateObjectWithAttributes<UniformRandomVariable>(
          "Min", DoubleValue(stratVmin), "Max", DoubleValue(stratVmax))),
      "SpeedNoise", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(stratSpeedVar))),
      "DirDevNoiseFar", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(stratDirFarVar))),
      "DirDevNoiseNear", PointerValue(CreateObjectWithAttributes<NormalRandomVariable>(
          "Mean", DoubleValue(0.0), "Variance", DoubleValue(stratDirNearVar)))
  );

  surveillanceUAV.SetPositionAllocator(initialPosition);
  surveillanceUAV.Install(uavNodes.Get(0));

  patrolUAV.SetPositionAllocator(initialPosition);
  patrolUAV.Install(uavNodes.Get(1));

  rapidUAV.SetPositionAllocator(initialPosition);
  rapidUAV.Install(uavNodes.Get(2));

  strategicUAV.SetPositionAllocator(initialPosition);
  strategicUAV.Install(uavNodes.Get(3));

  // EPC/NR Setup
  Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
  Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
  nrHelper->SetEpcHelper(epcHelper);

  nrHelper->SetUePhyAttribute("TxPower", DoubleValue(ueTxPowerDbm));
  nrHelper->SetGnbPhyAttribute("TxPower", DoubleValue(gnbTxPowerDbm));

  double centralFrequency = 3.5e9;
  double bandwidth = bwMHz * 1e6;

  CcBwpCreator ccBwpCreator;
  CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, 1);
  OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);
  BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

  Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
  channelHelper->SetAttribute("ChannelModel", StringValue("ThreeGpp"));
  channelHelper->SetAttribute("Scenario", StringValue("UMa"));
  channelHelper->AssignChannelsToBands({band});

  NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
  NetDeviceContainer ueDevs  = nrHelper->InstallUeDevice(uavNodes, allBwps);

  InternetStackHelper internet;
  internet.Install(uavNodes);

  Ipv4InterfaceContainer ueIfaces = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

  nrHelper->AttachToClosestGnb(ueDevs, gnbDevs);

  std::string rateStr = std::to_string(telemetryRateMbps) + "Mbps";

  // Remote Host
  Ptr<Node> pgw = epcHelper->GetPgwNode();

  NodeContainer remoteHostContainer;
  remoteHostContainer.Create(1);
  Ptr<Node> remoteHost = remoteHostContainer.Get(0);
  internet.Install(remoteHostContainer);

  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gbps")));
  p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));

  NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIfaces = ipv4h.Assign(internetDevices);

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
      ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
  remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

  Ipv4Address ueGateway = epcHelper->GetUeDefaultGatewayAddress();
  for (uint32_t i = 0; i < uavNodes.GetN(); ++i)
  {
    Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(uavNodes.Get(i)->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(ueGateway, 1);
  }

  std::vector<Ptr<IsacSensingApp3D>> sensingApps;
  sensingApps.reserve(numUAVs);

  for (uint32_t i = 0; i < numUAVs; ++i)
  {
    OnOffHelper onoff("ns3::UdpSocketFactory",
                      InetSocketAddress(internetIfaces.GetAddress(1), telemetryPort));
    onoff.SetAttribute("DataRate", DataRateValue(DataRate(rateStr)));
    onoff.SetAttribute("PacketSize", UintegerValue(telemetryPktSize));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));

    ApplicationContainer apps = onoff.Install(uavNodes.Get(i));
    apps.Start(Seconds(1.0 + 0.05*i));
    apps.Stop(Seconds(simulationTime - 0.1));
  }

  Ptr<IsacFusionReceiverApp3D> fusionRx = CreateObject<IsacFusionReceiverApp3D>();
  fusionRx->Configure(port, numUAVs, numCows, snrThresholdDb, nullptr);
  remoteHost->AddApplication(fusionRx);
  fusionRx->SetStartTime(Seconds(0.0));
  fusionRx->SetStopTime(Seconds(simulationTime - 0.5));

  PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), telemetryPort));
  ApplicationContainer sinkApps = sinkHelper.Install(remoteHost);
  sinkApps.Start(Seconds(0.0));
  sinkApps.Stop(Seconds(simulationTime - 0.1));

  // 3D Cow agent init with terrain
  std::vector<CowAgent3D> cows;
  cows.reserve(numCows);

  Ptr<UniformRandomVariable> urv = CreateObject<UniformRandomVariable>();
  Ptr<UniformRandomVariable> urvAng = CreateObject<UniformRandomVariable>();

  double half = areaSize / 2.0;

  for (uint32_t i = 0; i < numCows; ++i)
  {
    CowAgent3D c;
    c.id = i;
    c.tagId = 1000 + i;

    double cx = urv->GetValue(-0.9*half, 0.9*half);
    double cy = urv->GetValue(-0.9*half, 0.9*half);
    double cz = terrain.GetHeight(cx, cy);  // Get terrain height

    c.pos = Vector(cx, cy, cz);

    double ang = urvAng->GetValue(0.0, 2.0*M_PI);
    double spd = urv->GetValue(0.0, cowSpeedMax);
    c.vel = Vector(spd*std::cos(ang), spd*std::sin(ang), 0.0);

    cows.push_back(c);
  }

  fusionRx->SetCowTruthSource(&cows);

  // Cow update with terrain following
  std::function<void()> updateCows;
  updateCows = [&]() {
    for (auto &c : cows)
    {
      c.pos.x += c.vel.x * cowUpdateDt;
      c.pos.y += c.vel.y * cowUpdateDt;

      // Boundary reflection
      if (c.pos.x > half)  { c.pos.x = 2*half - c.pos.x; c.vel.x *= -1; }
      if (c.pos.x < -half) { c.pos.x = -2*half - c.pos.x; c.vel.x *= -1; }
      if (c.pos.y > half)  { c.pos.y = 2*half - c.pos.y; c.vel.y *= -1; }
      if (c.pos.y < -half) { c.pos.y = -2*half - c.pos.y; c.vel.y *= -1; }

      // Update z to follow terrain
      c.pos.z = terrain.GetHeight(c.pos.x, c.pos.y);
    }
    Simulator::Schedule(Seconds(cowUpdateDt), updateCows);
  };
  Simulator::Schedule(Seconds(0.0), updateCows);

  // Install 3D ISAC sensing apps
  double senseDts[4] = {survSenseDt, patSenseDt, rapidSenseDt, stratSenseDt};

  for (uint32_t i = 0; i < numUAVs; ++i)
  {
    Ptr<IsacSensingApp3D> app = CreateObject<IsacSensingApp3D>();
    app->Configure(uavNodes.Get(i), i, i,  // roleId = uavId for simplicity
                   &cows, &terrain,
                   internetIfaces.GetAddress(1), port,
                   Seconds(senseDts[i]),
                   snrThresholdDb,
                   sigmaPosMeters,
                   fc, ptDbm, gDb, rcsDbsm, noiseDbm,
                   areaSize);
    app->ConfigureProcessingEnergy(kappaProc, c0Proc, c1Proc, fProcHz);

    uavNodes.Get(i)->AddApplication(app);
    sensingApps.push_back(app);
    app->SetStartTime(Seconds(2.0 + 0.1*i));
    app->SetStopTime(Seconds(simulationTime - 0.5));

    UdpClientHelper telemetryClient(internetIfaces.GetAddress(1), telemetryPort);
    telemetryClient.SetAttribute("MaxPackets", UintegerValue(0));
    telemetryClient.SetAttribute("Interval", TimeValue(Seconds(telemetryInterval)));
    telemetryClient.SetAttribute("PacketSize", UintegerValue(telemetryPktSize));

    ApplicationContainer telemetryApps = telemetryClient.Install(uavNodes.Get(i));
    telemetryApps.Start(Seconds(1.0 + 0.1*i));
    telemetryApps.Stop(Seconds(simulationTime - 0.5));
  }

  // Energy accumulation
  std::vector<double> ePropJ(numUAVs, 0.0);
  std::vector<double> eRfJ(numUAVs, 0.0);

  double PhoverW = HoverPowerQuadrotor(uavMassKg, rotorRadiusM, rhoAir, etaProp, 4);
  double PtxUeW = DbmToW(ueTxPowerDbm);

  std::function<void()> sampleEnergy;
  sampleEnergy = [&]() {
    double dt = energySampleDt;

    for (uint32_t i = 0; i < numUAVs; ++i)
    {
      Ptr<MobilityModel> m = uavNodes.Get(i)->GetObject<MobilityModel>();
      Vector v = m->GetVelocity();
      double speed2 = v.x * v.x + v.y * v.y + v.z * v.z;

      double Pprop = PhoverW + cv * speed2;
      ePropJ[i] += Pprop * dt;

      eRfJ[i] += PtxUeW * dt;

      // Update role stats energy
      sensingApps[i]->GetRoleStats().energyJ = ePropJ[i] + eRfJ[i] + sensingApps[i]->GetProcessingEnergyJ();
    }

    Simulator::Schedule(Seconds(dt), sampleEnergy);
  };

  FlowMonitorHelper flowHelper;
  Ptr<FlowMonitor> monitor = flowHelper.Install(NodeContainer(uavNodes, remoteHostContainer));
  Ptr<Ipv4FlowClassifier> classifier =
      DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());

  Simulator::Schedule(Seconds(1.0), sampleEnergy);

  Simulator::Stop(Seconds(simulationTime));
  Simulator::Run();

  //   collection
  monitor->CheckForLostPackets();
  auto stats = monitor->GetFlowStats();

  double sumThr = 0.0, sumDelay = 0.0, sumLoss = 0.0;
  uint32_t nFlows = 0;

  std::cout << "\n===== APP (ISAC REPORTS + TELEMETRY) FLOW KPIs =====\n";
  for (const auto& kv : stats)
  {
    auto st = kv.second;
    auto t = classifier->FindFlow(kv.first);

    bool isIsac = (t.destinationPort == port);
    bool isTelem = (t.destinationPort == telemetryPort);
    if (!isIsac && !isTelem) continue;

    const char* tag = isIsac ? "[ISAC_REPORT]" : "[TELEMETRY]";

    double duration = (st.timeLastTxPacket - st.timeFirstTxPacket).GetSeconds();
    double thrMbps = (duration > 0.0) ? (st.rxBytes * 8.0 / duration / 1e6) : 0.0;
    double avgDelayMs = (st.rxPackets > 0)
        ? (st.delaySum.GetSeconds() / st.rxPackets * 1000.0)
        : 0.0;
    double lossPct = (st.txPackets > 0)
        ? (100.0 * double(st.txPackets - st.rxPackets) / double(st.txPackets))
        : 0.0;

    if (isIsac)
    {
      sumThr += thrMbps;
      sumDelay += avgDelayMs;
      sumLoss += lossPct;
      nFlows++;
    }

    std::cout << tag << " Flow " << t.sourceAddress << " -> " << t.destinationAddress
              << " Tx=" << st.txPackets
              << " Rx=" << st.rxPackets
              << " Loss%=" << lossPct
              << " AvgDelay(ms)=" << avgDelayMs
              << " Throughput(Mbps)=" << thrMbps
              << "\n";
  }

  double avgThr = (nFlows > 0) ? (sumThr / nFlows) : 0.0;
  double avgDelay = (nFlows > 0) ? (sumDelay / nFlows) : 0.0;
  double avgLoss = (nFlows > 0) ? (sumLoss / nFlows) : 0.0;

  double runPdet = fusionRx->GetRunPdet();
  double runRmse = fusionRx->GetRunRmseM();

  // Per-role metrics
  const char* roleNames[4] = {"Surveillance", "Patrol", "Rapid", "Strategic"};

  std::cout << "\n===== PER-ROLE METRICS =====\n";
  for (uint32_t r = 0; r < 4; ++r)
  {
    const auto& rs = sensingApps[r]->GetRoleStats();
    double rolePdet = fusionRx->GetRolePdet(r);
    double roleRmse = fusionRx->GetRoleRmseM(r);
    double roleEnergy = rs.energyJ;
    double energyPerDet = (rs.detections > 0) ? (roleEnergy / rs.detections) : 999999.0;
    double detRate = (simulationTime > 0) ? (double(rs.detections) / simulationTime) : 0.0;
    double uniqueCows = rs.uniqueCowsDetected.size();
    double sensingDuty = (simulationTime > 0) ? (rs.sensingTimeS / simulationTime) : 0.0;
    double dataPerDet = (rs.detections > 0) ? (double(rs.dataVolumeBytes) / rs.detections) : 0.0;

    // Area coverage
    double cellSize = 20.0;
    double totalCells = (areaSize / cellSize) * (areaSize / cellSize);
    double coverageRatio = (totalCells > 0) ? (double(rs.coveredCells.size()) / totalCells) : 0.0;

    std::cout << roleNames[r] << " (UAV " << r << "):\n"
              << "  Detections: " << rs.detections << "\n"
              << "  Det Rate: " << detRate << " det/s\n"
              << "  Pdet: " << (rolePdet * 100.0) << "%\n"
              << "  RMSE 3D: " << roleRmse << " m\n"
              << "  Energy: " << roleEnergy << " J\n"
              << "  Energy/Det: " << energyPerDet << " J/det\n"
              << "  Unique Cows: " << uniqueCows << "\n"
              << "  Coverage: " << (coverageRatio * 100.0) << "%\n"
              << "  Sensing Duty: " << (sensingDuty * 100.0) << "%\n"
              << "  Data/Det: " << dataPerDet << " bytes\n";
  }

  // Energy totals
  double ePropTot = 0.0, eRfTot = 0.0, eProcTot = 0.0, eTotTot = 0.0;

  std::cout << "\n===== PER-UAV ENERGY =====\n";
  for (uint32_t i = 0; i < numUAVs; ++i)
  {
    double eProc = sensingApps[i]->GetProcessingEnergyJ();
    double eTot = ePropJ[i] + eRfJ[i] + eProc;

    ePropTot += ePropJ[i];
    eRfTot += eRfJ[i];
    eProcTot += eProc;

    std::cout << "UAV " << i
              << " | Eprop=" << ePropJ[i]
              << " J | ERF=" << eRfJ[i]
              << " J | Eproc=" << eProc
              << " J | Etot=" << eTot
              << " J\n";
  }

  eTotTot = ePropTot + eRfTot + eProcTot;

  // Jfinal calculation (unchanged)
  const double thrRefMbps   = 0.0020;
  const double rmseRefM     = 30.0;
  const double delayRefMs   = 20.0;
  const double energyRefJ   = 200000.0;

  double Pdet_norm   = ClampToRange(runPdet, 0.0, 1.0);
  double Thr_norm    = ClampToRange(avgThr / thrRefMbps, 0.0, 1.0);
  double RMSE_norm   = ClampToRange(runRmse / rmseRefM, 0.0, 1.0);
  double Delay_norm  = ClampToRange(avgDelay / delayRefMs, 0.0, 1.0);
  double Loss_norm   = ClampToRange(avgLoss / 100.0, 0.0, 1.0);
  double Energy_norm = ClampToRange(eTotTot / energyRefJ, 0.0, 1.0);

  double Jfinal =
      3.0 * Pdet_norm
    + 3.0 * Thr_norm
    - 2.0 * RMSE_norm
    - 1.0 * Delay_norm
    - 1.0 * Loss_norm;

  // Save summary CSV with 3D suffix
  bool needHeader = FileEmptyOrMissing("run_summary_3d.csv");
  std::ofstream runSummary("run_summary_3d.csv", std::ios::app);

  if (needHeader)
  {
    runSummary
      << "rngRun,numCows,snrThDb,sigmaPosM,"
      << "ueTxPowerDbm,gnbTxPowerDbm,cv,"
      << "survSenseDt,patSenseDt,rapidSenseDt,stratSenseDt,"
      << "Pdet,RMSEm_3D,Jfinal,"
      << "alphaSurv,alphaPat,alphaRapid,alphaStrat,"
      << "survVmin,survVmax,patVmin,patVmax,rapidVmin,rapidVmax,stratVmin,stratVmax,"
      << "survZmin,survZmax,patZmin,patZmax,rapidZmin,rapidZmax,stratZmin,stratZmax,"
      << "survDt,patDt,rapidDt,stratDt,"
      << "avgThrMbps,avgDelayMs,avgLossPct,"
      << "UAV0_Energy_J,UAV1_Energy_J,UAV2_Energy_J,UAV3_Energy_J,"
      << "EpropJ,ERFJ,EprocJ,EtotJ,"
      // New per-role KPIs
      << "survDetections,patDetections,rapidDetections,stratDetections,"
      << "survDetRate,patDetRate,rapidDetRate,stratDetRate,"
      << "survRMSE,patRMSE,rapidRMSE,stratRMSE,"
      << "survEnergyPerDet,patEnergyPerDet,rapidEnergyPerDet,stratEnergyPerDet,"
      << "survCoverage,patCoverage,rapidCoverage,stratCoverage,"
      << "survSensingDuty,patSensingDuty,rapidSensingDuty,stratSensingDuty,"
      << "survDataPerDet,patDataPerDet,rapidDataPerDet,stratDataPerDet,"
      << "terrainHillHeight,terrainHillFreq\n";
  }

  double uav0Etot = ePropJ[0] + eRfJ[0] + sensingApps[0]->GetProcessingEnergyJ();
  double uav1Etot = ePropJ[1] + eRfJ[1] + sensingApps[1]->GetProcessingEnergyJ();
  double uav2Etot = ePropJ[2] + eRfJ[2] + sensingApps[2]->GetProcessingEnergyJ();
  double uav3Etot = ePropJ[3] + eRfJ[3] + sensingApps[3]->GetProcessingEnergyJ();

  // Per-role stats for CSV
  std::vector<double> roleDetections(4), roleDetRates(4), roleRmses(4);
  std::vector<double> roleEnergyPerDet(4), roleCoverage(4), roleSensingDuty(4), roleDataPerDet(4);

  double cellSize = 20.0;
  double totalCells = (areaSize / cellSize) * (areaSize / cellSize);

  for (uint32_t r = 0; r < 4; ++r)
  {
    const auto& rs = sensingApps[r]->GetRoleStats();
    roleDetections[r] = rs.detections;
    roleDetRates[r] = (simulationTime > 0) ? (double(rs.detections) / simulationTime) : 0.0;
    roleRmses[r] = fusionRx->GetRoleRmseM(r);
    roleEnergyPerDet[r] = (rs.detections > 0) ? (rs.energyJ / rs.detections) : 999999.0;
    roleCoverage[r] = (totalCells > 0) ? (double(rs.coveredCells.size()) / totalCells) : 0.0;
    roleSensingDuty[r] = (simulationTime > 0) ? (rs.sensingTimeS / simulationTime) : 0.0;
    roleDataPerDet[r] = (rs.detections > 0) ? (double(rs.dataVolumeBytes) / rs.detections) : 0.0;
  }

  runSummary
    << rngRun << ","
    << numCows << ","
    << snrThresholdDb << ","
    << sigmaPosMeters << ","
    << ueTxPowerDbm << ","
    << gnbTxPowerDbm << ","
    << cv << ","
    << survSenseDt << ","
    << patSenseDt << ","
    << rapidSenseDt << ","
    << stratSenseDt << ","
    << runPdet << ","
    << runRmse << ","
    << Jfinal << ","
    << alphaSurv << ","
    << alphaPat << ","
    << alphaRapid << ","
    << alphaStrat << ","
    << survVmin << "," << survVmax << ","
    << patVmin  << "," << patVmax  << ","
    << rapidVmin << "," << rapidVmax << ","
    << stratVmin << "," << stratVmax << ","
    << survZmin << "," << survZmax << ","
    << patZmin  << "," << patZmax  << ","
    << rapidZmin << "," << rapidZmax << ","
    << stratZmin << "," << stratZmax << ","
    << survDt << ","
    << patDt << ","
    << rapidDt << ","
    << stratDt << ","
    << avgThr << ","
    << avgDelay << ","
    << avgLoss << ","
    << uav0Etot << ","
    << uav1Etot << ","
    << uav2Etot << ","
    << uav3Etot << ","
    << ePropTot << ","
    << eRfTot << ","
    << eProcTot << ","
    << eTotTot << ","
    // Per-role KPIs
    << roleDetections[0] << "," << roleDetections[1] << "," << roleDetections[2] << "," << roleDetections[3] << ","
    << roleDetRates[0] << "," << roleDetRates[1] << "," << roleDetRates[2] << "," << roleDetRates[3] << ","
    << roleRmses[0] << "," << roleRmses[1] << "," << roleRmses[2] << "," << roleRmses[3] << ","
    << roleEnergyPerDet[0] << "," << roleEnergyPerDet[1] << "," << roleEnergyPerDet[2] << "," << roleEnergyPerDet[3] << ","
    << roleCoverage[0] << "," << roleCoverage[1] << "," << roleCoverage[2] << "," << roleCoverage[3] << ","
    << roleSensingDuty[0] << "," << roleSensingDuty[1] << "," << roleSensingDuty[2] << "," << roleSensingDuty[3] << ","
    << roleDataPerDet[0] << "," << roleDataPerDet[1] << "," << roleDataPerDet[2] << "," << roleDataPerDet[3] << ","
    << terrainHillHeight << "," << terrainHillFreq
    << "\n";

  runSummary.close();

  Simulator::Destroy();
  std::cout << "\nFINAL_PDET=" << runPdet << "\n";
  std::cout << "FINAL_RMSE_3D=" << runRmse << "\n";
  std::cout << "FINAL_J=" << Jfinal << "\n";
  std::cout << "\nDone (3D version).\n";
  std::cout << "Generated: run_summary_3d.csv\n";

  return 0;
}
