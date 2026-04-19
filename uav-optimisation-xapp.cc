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
#include <limits>
#include <cstring>
#include <sstream>
#include <string>
#include <sys/socket.h>   // ADD
#include <arpa/inet.h>    // ADD
#include <unistd.h>       // ADD
#include <fcntl.h>        // ADD

using namespace ns3;

// ============================================================
// KpiReporter: bridges ns-3 to the xApp via UDP sockets
// Sends KPI reports to xApp; receives param updates back
// ============================================================
class KpiReporter
{
public:
  KpiReporter(const std::string& xappHost,
              uint16_t sendPort,
              uint16_t recvPort)
  {
    // Send socket: KPIs → xApp
    m_sendSock = socket(AF_INET, SOCK_DGRAM, 0);
    m_sendAddr.sin_family = AF_INET;
    m_sendAddr.sin_port   = htons(sendPort);
    inet_pton(AF_INET, xappHost.c_str(), &m_sendAddr.sin_addr);

    // Receive socket: param updates ← xApp (non-blocking)
    m_recvSock = socket(AF_INET, SOCK_DGRAM, 0);
    int flags  = fcntl(m_recvSock, F_GETFL, 0);
    fcntl(m_recvSock, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in recvAddr{};
    recvAddr.sin_family      = AF_INET;
    recvAddr.sin_port        = htons(recvPort);
    recvAddr.sin_addr.s_addr = INADDR_ANY;
    bind(m_recvSock,
         reinterpret_cast<struct sockaddr*>(&recvAddr),
         sizeof(recvAddr));

    std::cout << "[KpiReporter] Initialised — send→"
              << xappHost << ":" << sendPort
              << "  recv←" << recvPort << "\n";
  }

  ~KpiReporter()
  {
    close(m_sendSock);
    close(m_recvSock);
  }

  void SendKpis(double pdet,     double rmse,
                double energyJ,  double simTime,
                double thrMbps,  double delayMs,
                double lossPct)
  {
    std::ostringstream ss;
    ss << "{"
       << "\"pdet\":"       << pdet     << ","
       << "\"rmse_m\":"     << rmse     << ","
       << "\"energy_j\":"   << energyJ  << ","
       << "\"sim_time\":"   << simTime  << ","
       << "\"throughput\":" << thrMbps  << ","
       << "\"delay_ms\":"   << delayMs  << ","
       << "\"loss_pct\":"   << lossPct
       << "}";
    std::string msg = ss.str();
    sendto(m_sendSock,
           msg.c_str(), msg.size(), 0,
           reinterpret_cast<struct sockaddr*>(&m_sendAddr),
           sizeof(m_sendAddr));
    std::cout << "[KpiReporter] t=" << simTime
              << "s  sent: " << msg << "\n";
  }

  // Returns true if xApp sent a param update
  bool ReceiveParamUpdate(std::map<std::string, double>& params)
  {
    char buf[4096];
    ssize_t n = recv(m_recvSock, buf, sizeof(buf) - 1, 0);
    if (n <= 0) return false;
    buf[n] = '\0';
    std::string msg(buf);
    std::cout << "[KpiReporter] Param update: " << msg << "\n";

    auto extractDouble = [&](const std::string& key) -> double {
      std::string search = "\"" + key + "\":";
      size_t pos = msg.find(search);
      if (pos == std::string::npos) return -1.0;
      pos += search.size();
      size_t end = msg.find_first_of(",}", pos);
      if (end == std::string::npos) end = msg.size();
      try { return std::stod(msg.substr(pos, end - pos)); }
      catch (...) { return -1.0; }
    };

    // All parameters that the xApp can update
    std::vector<std::string> keys = {
      "survSenseDt", "patSenseDt", "rapidSenseDt", "stratSenseDt",
      "survVmax",    "patVmax",    "rapidVmax",    "stratVmax",
      "survZmin",    "survZmax",   "patZmin",      "patZmax",
      "rapidZmin",   "rapidZmax",  "stratZmin",    "stratZmax",
      "survDt",      "patDt",      "rapidDt",      "stratDt"
    };

    bool anyUpdate = false;
    for (const auto& key : keys)
    {
      double v = extractDouble(key);
      if (v >= 0.0) { params[key] = v; anyUpdate = true; }
    }
    return anyUpdate;
  }

private:
  int m_sendSock;
  int m_recvSock;
  struct sockaddr_in m_sendAddr{};
};

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

/* ============================================================================
 *  YOUR EnhancedGaussMarkovMobilityModel CLASS (unchanged)
 *  (Paste your full class here exactly as you have it)
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

  // Allow external control (e.g., future xApp / DRL logic) without rewriting attributes:
  void SetAlpha (double a) { m_alpha = std::clamp(a, 0.0, 1.0); }
  void SetMeanSpeedValue (double ms) { m_meanSpeed = std::max(0.0, ms); }

  // NEW: target altitude control
  void SetTargetAltitude (double z)
  {
    m_targetAltitude = std::max(m_bounds.zMin, std::min(z, m_bounds.zMax));
  }

  double GetTargetAltitude () const
  {
    return m_targetAltitude;
  }

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

    // NEW: initialise target altitude to current altitude
    m_targetAltitude = std::max(m_bounds.zMin, std::min(m_position.z, m_bounds.zMax));

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

    // NEW: smooth altitude tracking toward target altitude
    double maxVz = 2.0; // m/s vertical rate limit
    double zStep = maxVz * dt;
    double dz = m_targetAltitude - m_position.z;
    dz = std::max(-zStep, std::min(dz, zStep));
    next.z = m_position.z + dz;

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

  double m_speed;          // current speed
  double m_dir;            // current direction (heading) radians
  double m_dirDev;         // current direction deviation radians
  double m_meanSpeed;      // sampled once
  double m_targetAltitude; // NEW: DRL-controlled altitude target

  EventId m_event;
};

NS_OBJECT_ENSURE_REGISTERED (EnhancedGaussMarkovMobilityModel);
// ============================================================================
// Cow agents (Option 1): NOT ns-3 nodes
// ============================================================================
struct IsacMeas
{
  bool detected{false};
  double snrDb{-1e9};
  double x{std::numeric_limits<double>::quiet_NaN()};
  double y{std::numeric_limits<double>::quiet_NaN()};
  uint32_t tagId{0};
};

static bool FuseSNRWeighted(const std::vector<IsacMeas>& meas,
                            double snrThDb,
                            double& xFused,
                            double& yFused,
                            uint32_t& nUsed)
{
  nUsed = 0;
  double sumW = 0.0;
  double sumX = 0.0;
  double sumY = 0.0;

  for (const auto& m : meas)
  {
    if (!m.detected) continue;
    if (!std::isfinite(m.x) || !std::isfinite(m.y)) continue;
    if (m.snrDb < snrThDb) continue;

    double w = std::pow(10.0, m.snrDb / 10.0); // dB -> linear
    if (!std::isfinite(w) || w <= 0) continue;

    sumW += w;
    sumX += w * m.x;
    sumY += w * m.y;
    nUsed++;
  }

  if (nUsed == 0 || sumW <= 0.0)
  {
    xFused = std::numeric_limits<double>::quiet_NaN();
    yFused = std::numeric_limits<double>::quiet_NaN();
    return false;
  }

  xFused = sumX / sumW;
  yFused = sumY / sumW;
  return true;
}


struct CowAgent
{
  uint32_t id;
  Vector pos;     // (x,y,0)
  Vector vel;     // (vx,vy,0)
  uint32_t tagId; // simple RFID/tag identifier
};

// simple helper
static double Clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

// ============================================================================
// ISAC Sensing Application: runs on each UAV (UE) and sends sensing reports
// ============================================================================
class IsacSensingApp : public Application
{
public:
  IsacSensingApp() = default;

  void Configure(Ptr<Node> uav,
                 uint32_t uavId,
                 std::vector<CowAgent>* cows,
                 Ipv4Address remoteAddr,
                 uint16_t remotePort,
                 Time sensingInterval,
                 double snrThresholdDb,
                 double sigmaPosMeters,
                 double fcHz,
                 double ptDbm,
                 double gDb,
                 double rcsDbsm,
                 double noiseDbm)
  {
    m_uav = uav;
    m_uavId = uavId;
    m_cows = cows;
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

    // precompute constants
    m_c = 3e8;
    m_lambda = m_c / m_fc;

    m_ptW = std::pow(10.0, (m_ptDbm - 30.0) / 10.0);
    m_gLin = std::pow(10.0, (m_gDb) / 10.0);
    m_rcsLin = std::pow(10.0, (m_rcsDbsm) / 10.0);
    m_noiseW = std::pow(10.0, (m_noiseDbm - 30.0) / 10.0);

    m_rng = CreateObject<NormalRandomVariable>();
    m_rng->SetAttribute("Mean", DoubleValue(0.0));
    m_rng->SetAttribute("Variance", DoubleValue(m_sigmaPos * m_sigmaPos));
  }

  void ConfigureProcessingEnergy(double kappa, double c0, double c1, double fProcHz)
  {
    m_kappa = kappa;
    m_c0 = c0;
    m_c1 = c1;
    m_fProcHz = fProcHz;
  }

  double GetProcessingEnergyJ() const
  {
    return m_eProcJ;
  }

  // NEW: allow online DRL control of sensing interval
 

  double GetSensingIntervalSeconds() const
  {
    return m_sensingInterval.GetSeconds();
  }
  
void SetSensingInterval(double newIntervalS)
{
  m_sensingInterval = Seconds(std::max(0.3, newIntervalS));
}

  void SetLogFiles(std::ofstream* reportCsv) { m_reportCsv = reportCsv; }

private:
  void StartApplication() override
  {
    // UDP socket
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
    m_sendEvent = Simulator::Schedule(m_sensingInterval, &IsacSensingApp::SenseAndSend, this);
  }

  // --- Core sensing model ---
  // Monostatic radar-like received power (lumped):
  //   Pr = Pt * G^2 * lambda^2 * sigma / ((4*pi)^3 * R^4)
  // SNR = Pr / N0
  double ComputeSnrDb(double R) const
  {
    double denom = (std::pow(4.0 * M_PI, 3.0) * std::pow(R, 4.0)) + 1e-30;
    double pr = (m_ptW * (m_gLin * m_gLin) * (m_lambda * m_lambda) * m_rcsLin) / denom;
    double snrLin = pr / std::max(m_noiseW, 1e-30);
    return 10.0 * std::log10(std::max(snrLin, 1e-12));
  }

  void SenseAndSend()
  {
    Ptr<MobilityModel> mob = m_uav->GetObject<MobilityModel>();
    Vector u = mob->GetPosition(); // UAV position

    // Packet payload format (per detection):
    // [uavId(uint16)] [numDet(uint16)] then repeated:
    // [cowId(uint16)] [tagId(uint16)] [xhat(float)] [yhat(float)] [snr(float)]
    // This is intentionally simple + small.

    std::vector<uint8_t> buf;
    auto push_u16 = [&](uint16_t v) {
      buf.push_back(uint8_t(v & 0xFF));
      buf.push_back(uint8_t((v >> 8) & 0xFF));
    };
    auto push_f32 = [&](float f) {
      uint32_t x;
      static_assert(sizeof(float) == 4, "float must be 4 bytes");
      std::memcpy(&x, &f, 4);
      buf.push_back(uint8_t(x & 0xFF));
      buf.push_back(uint8_t((x >> 8) & 0xFF));
      buf.push_back(uint8_t((x >> 16) & 0xFF));
      buf.push_back(uint8_t((x >> 24) & 0xFF));
    };

    uint16_t numDet = 0;
    // reserve header
    push_u16((uint16_t)m_uavId);
    push_u16(0); // placeholder for numDet

    double t = Simulator::Now().GetSeconds();

    for (auto& cow : *m_cows)
    {
      Vector cpos = cow.pos; // (x,y,0)
      double R = CalculateDistance(u, Vector(cpos.x, cpos.y, u.z)); // 3D using UAV altitude
      double snrDb = ComputeSnrDb(R);

      bool detected = (snrDb >= m_snrThDb);

      // Add detection variability naturally:
      // If near threshold, sometimes miss (soft detection). This prevents "always 100%".
      if (detected)
      {
        double margin = snrDb - m_snrThDb;
        if (margin < 6.0) // within 6 dB of threshold -> stochastic misses
        {
          double p = Clamp(margin / 6.0, 0.0, 1.0); // 0..1
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
          (*m_reportCsv) << t << "," << m_uavId << "," << cow.id << "," << cow.tagId
                         << ",0," << snrDb << ",nan,nan\n";
        }
        continue;
      }

      // Position estimate = true + Gaussian noise (MVP)
      double xhat = cow.pos.x + m_rng->GetValue();
      double yhat = cow.pos.y + m_rng->GetValue();

      // append detection entry
      push_u16((uint16_t)cow.id);
      push_u16((uint16_t)cow.tagId);
      push_f32((float)xhat);
      push_f32((float)yhat);
      push_f32((float)snrDb);

      numDet++;

      if (m_reportCsv)
      {
        (*m_reportCsv) << t << "," << m_uavId << "," << cow.id << "," << cow.tagId
                       << ",1," << snrDb << "," << xhat << "," << yhat << "\n";
      }
    }

    // patch numDet into header bytes [2..3]
    buf[2] = uint8_t(numDet & 0xFF);
    buf[3] = uint8_t((numDet >> 8) & 0xFF);

    Ptr<Packet> p = Create<Packet>(buf.data(), buf.size());
    m_socket->Send(p);
// ==================== PROCESSING ENERGY ACCUMULATION ====================
    // E_proc_step = kappa * (c0 + c1 * numDet) * f^2
    double eProcStepJ = m_kappa * (m_c0 + m_c1 * static_cast<double>(numDet)) * m_fProcHz * m_fProcHz;
    m_eProcJ += eProcStepJ;
    ScheduleNext();
  }

private:
  Ptr<Node> m_uav;
  uint32_t m_uavId = 0;
  std::vector<CowAgent>* m_cows = nullptr;

  Ptr<Socket> m_socket;
  bool m_running = false;
  EventId m_sendEvent;

  Ipv4Address m_remoteAddr;
  uint16_t m_remotePort = 9000;
  Time m_sensingInterval = Seconds(1.0);

  // sensing params
  double m_snrThDb = 30.0;
  double m_sigmaPos = 5.0;

  // radar/link params
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

  Ptr<NormalRandomVariable> m_rng;

  std::ofstream* m_reportCsv = nullptr; // optional CSV logging

  // ==================== PROCESSING ENERGY MODEL ====================
  double m_kappa = 1e-27;
  double m_c0 = 1e7;      // baseline CPU cycles per sensing step
  double m_c1 = 5e6;      // extra CPU cycles per detected target
  double m_fProcHz = 1e9; // processor frequency (Hz)

  double m_eProcJ = 0.0;  // accumulated processing energy (J)
};

class IsacFusionReceiverApp : public Application
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

    m_windowDetected = 0;
    m_windowTotal = 0;
    m_windowSqErrSum = 0.0;
    m_windowRmseCount = 0;
  }

  void SetCowTruthSource(std::vector<CowAgent>* cows)
  {
    m_cows = cows;
  }

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

  void ResetWindowStats()
  {
    m_windowDetected = 0;
    m_windowTotal = 0;
    m_windowSqErrSum = 0.0;
    m_windowRmseCount = 0;
    
  }

private:
  void StartApplication() override
  {
    m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
    InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
    m_socket->Bind(local);
    m_socket->SetRecvCallback(MakeCallback(&IsacFusionReceiverApp::HandleRead, this));
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

      if (size < 4) return;

      uint16_t uavId = ReadU16(&data[0]);
      uint16_t numDet = ReadU16(&data[2]);

      uint32_t offset = 4;
      double t = Simulator::Now().GetSeconds();

      for (uint16_t i = 0; i < numDet; ++i)
      {
        if (offset + 2 + 2 + 4 + 4 + 4 > size) break;

        uint16_t cowId = ReadU16(&data[offset]); offset += 2;
        uint16_t tagId = ReadU16(&data[offset]); offset += 2;
        float xhat = ReadF32(&data[offset]); offset += 4;
        float yhat = ReadF32(&data[offset]); offset += 4;
        float snr  = ReadF32(&data[offset]); offset += 4;

        if (cowId >= m_numCows || uavId >= m_numUavs) continue;

        IsacMeas m;
        m.detected = true;
        m.snrDb = snr;
        m.x = xhat;
        m.y = yhat;
        m.tagId = tagId;

        m_meas[cowId][uavId] = m;
      }

      for (uint32_t cowId = 0; cowId < m_numCows; ++cowId)
{
    std::vector<IsacMeas> vec;
    vec.reserve(m_numUavs);
    for (uint32_t u = 0; u < m_numUavs; ++u) vec.push_back(m_meas[cowId][u]);

    double xf, yf;
    uint32_t nUsed;
    bool detF = FuseSNRWeighted(vec, m_snrThDb, xf, yf, nUsed);

    uint32_t tagId = (vec.size() > 0 ? vec[0].tagId : (1000 + cowId));

    if (m_fusedCsv)
    {
        (*m_fusedCsv) << t << "," << cowId << "," << tagId
                      << "," << (detF ? 1 : 0)
                      << "," << nUsed
                      << "," << xf << "," << yf << "\n";
    }

    m_windowTotal++;
    m_runTotal++;

    if (detF)
    {
        m_windowDetected++;
        m_runDetected++;

        if (m_cows && cowId < m_cows->size())
        {
            double dx = xf - (*m_cows)[cowId].pos.x;
            double dy = yf - (*m_cows)[cowId].pos.y;
            double se = dx * dx + dy * dy;

            m_windowSqErrSum += se;
            m_windowRmseCount++;

            m_runSqErrSum += se;
            m_runRmseCount++;
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
  std::vector<CowAgent>* m_cows{nullptr};

  std::vector<std::vector<IsacMeas>> m_meas;

  uint32_t m_windowDetected{0};
  uint32_t m_windowTotal{0};
  double m_windowSqErrSum{0.0};
  uint32_t m_windowRmseCount{0};
  uint32_t m_runDetected{0};
  uint32_t m_runTotal{0};
  double m_runSqErrSum{0.0};
  uint32_t m_runRmseCount{0};
};

// ============================================================================
// DRL / Online Control Helpers
// ============================================================================


// ============================================================================
// Main
// ============================================================================
bool FileEmptyOrMissing(const std::string& name)
{
  struct stat st;
  if (stat(name.c_str(), &st) != 0) return true;   // missing
  return (st.st_size == 0);                        // empty
}
static double ClampToRange(double v, double lo, double hi)
{
  return std::max(lo, std::min(v, hi));
}

int main(int argc, char *argv[])
{
  Time::SetResolution(Time::NS);

  // --- Sim parameters ---
  uint32_t numUAVs = 4;
  double simulationTime = 300.0;
  double areaSize = 800.0;
  uint16_t port = 9000;
  double bwMHz = 20.0; // default bandwidth (MHz)
  

// Telemetry load (background traffic) - Option A
double telemetryRateMbps = 5.0; // per UAV (Mbps)


  // ==================== EXTRA COMM (telemetry) FLOW KNOBS ====================
uint16_t telemetryPort = 9100;     // different from sensing port (9000)
uint32_t telemetryPktSize = 1200;   // bytes
double telemetryInterval = 0.1;    // seconds (0.1s = 10 packets/sec)

 // ==================== OPTIMISATION KNOBS (Mobility decision variables) ====================

// gNB altitude
double gnbZ = 50.0;

// Altitude ranges for each UAV role (zMin/zMax in Bounds)
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

// Speeds (Uniform RV bounds)
double survVmin = 8.0,  survVmax = 15.0;
double patVmin  = 15.0, patVmax  = 25.0;
double rapidVmin= 20.0, rapidVmax= 35.0;
double stratVmin= 12.0, stratVmax= 22.0;

// Common boundary / turn limits
double margin = 80.0;

double survMaxTurnBiasDeg = 22.5;
double patMaxTurnBiasDeg  = 22.5;
double rapidMaxTurnBiasDeg= 25.0;
double stratMaxTurnBiasDeg= 20.0;

double survMaxDevStepDeg = 8.0;
double patMaxDevStepDeg  = 10.0;
double rapidMaxDevStepDeg= 12.0;
double stratMaxDevStepDeg= 8.0;

// Noise variances (Normal RV variance)
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
double uav0z = 60, uav1z = 80, uav2z = 100, uav3z = 120; // initial altitudes


  // ISAC params (tune)
  uint32_t numCows = 80;
  double cowSpeedMax = 1.0;          // m/s
  double cowUpdateDt = 1.0;          // seconds
  double survSenseDt  = 1.0;   // seconds
double patSenseDt   = 1.0;
double rapidSenseDt = 1.0;
double stratSenseDt = 1.0;      // seconds
  double snrThresholdDb = 30.0;      // tune for variation (try 25..45)
  double sigmaPosMeters = 5.0;       // measurement noise (position estimate std dev)

  // Radar-ish params (starter values like your MATLAB MVP)
  double fc = 3.5e9;
  double ptDbm = 30.0;
  double gDb = 20.0;
  double rcsDbsm = -5.0;
  double noiseDbm = -110.0;

  double ueTxPowerDbm = 23.0;
  double gnbTxPowerDbm = 38.0;

// ==================== ENERGY MODEL PARAMS ====================
  double uavMassKg = 2.8;
  double rotorRadiusM = 0.15;
  double rhoAir = 1.225;
  double etaProp = 0.7;
  double cv = 0.1;            // propulsion coefficient for speed^2 term
  double energySampleDt = 1.0;

  // Processing-energy parameters
  double kappaProc = 1e-27;
  double c0Proc = 1e7;
  double c1Proc = 5e6;
  double fProcHz = 1e9;

  CommandLine cmd(__FILE__);
uint32_t rngRun = 1;

// ================= CORE SIM =================
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


// ================= MOBILITY: ALPHAS =================
cmd.AddValue("alphaSurv",  "Alpha for Surveillance UAV", alphaSurv);
cmd.AddValue("alphaPat",   "Alpha for Patrol UAV", alphaPat);
cmd.AddValue("alphaRapid", "Alpha for Rapid UAV", alphaRapid);
cmd.AddValue("alphaStrat", "Alpha for Strategic UAV", alphaStrat);


// ================= MOBILITY: SPEED RANGES =================
cmd.AddValue("survVmin", "Surveillance min speed (m/s)", survVmin);
cmd.AddValue("survVmax", "Surveillance max speed (m/s)", survVmax);

cmd.AddValue("patVmin",  "Patrol min speed (m/s)", patVmin);
cmd.AddValue("patVmax",  "Patrol max speed (m/s)", patVmax);

cmd.AddValue("rapidVmin","Rapid min speed (m/s)", rapidVmin);
cmd.AddValue("rapidVmax","Rapid max speed (m/s)", rapidVmax);

cmd.AddValue("stratVmin","Strategic min speed (m/s)", stratVmin);
cmd.AddValue("stratVmax","Strategic max speed (m/s)", stratVmax);


// ================= MOBILITY: ALTITUDE BOUNDS =================
cmd.AddValue("survZmin","Surveillance zMin",survZmin);
cmd.AddValue("survZmax","Surveillance zMax",survZmax);

cmd.AddValue("patZmin","Patrol zMin",patZmin);
cmd.AddValue("patZmax","Patrol zMax",patZmax);

cmd.AddValue("rapidZmin","Rapid zMin",rapidZmin);
cmd.AddValue("rapidZmax","Rapid zMax",rapidZmax);

cmd.AddValue("stratZmin","Strategic zMin",stratZmin);
cmd.AddValue("stratZmax","Strategic zMax",stratZmax);


// ================= MOBILITY: TIMESTEPS =================
cmd.AddValue("survDt","Surveillance timestep (s)",survDt);
cmd.AddValue("patDt","Patrol timestep (s)",patDt);
cmd.AddValue("rapidDt","Rapid timestep (s)",rapidDt);
cmd.AddValue("stratDt","Strategic timestep (s)",stratDt);


cmd.AddValue("telemetryPktSize", "Telemetry packet size (bytes)", telemetryPktSize);
cmd.AddValue("telemetryInterval", "Telemetry packet interval (s)", telemetryInterval);
cmd.AddValue("telemetryPort", "Telemetry UDP port", telemetryPort);
cmd.AddValue("bwMHz", "NR bandwidth in MHz", bwMHz);


cmd.AddValue("telemetryRateMbps", "Background telemetry rate per UAV (Mbps)", telemetryRateMbps);
cmd.AddValue("telemetryPktSize", "Telemetry packet size (bytes)", telemetryPktSize);
cmd.AddValue("telemetryPort", "Telemetry UDP port", telemetryPort);

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

// ── ADD THESE TWO LINES ───────────────────────────────────
bool        enableXapp = false;
std::string xappHost   = "127.0.0.1";
cmd.AddValue("enableXapp", "Enable xApp UDP bridge (0/1)", enableXapp);
cmd.AddValue("xappHost",   "xApp host IP",                 xappHost);
// ─────────────────────────────────────────────────────────

cmd.Parse(argc, argv);
RngSeedManager::SetSeed(1);
RngSeedManager::SetRun(rngRun);

  // --- Nodes: UAVs (UEs) + 1 gNB + remote host ---
  NodeContainer uavNodes;
  uavNodes.Create(numUAVs);

  NodeContainer gnbNodes;
  gnbNodes.Create(1);

  // ==================== gNB mobility ====================
MobilityHelper gnbMobility;
gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
gnbMobility.Install(gnbNodes);
gnbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0.0, 0.0, gnbZ));

// ==================== UAV mobility patterns ====================

// Pattern 1: Surveillance UAV (Slow, High Persistence)
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

// Pattern 2: Patrol UAV (Medium Speed, Balanced)
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

// Pattern 3: Rapid Response UAV (Fast, More Agile)
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

// Pattern 4: Strategic UAV (Stable, Wider sweep)
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

  // ==================== INITIAL POSITIONS ====================
  Ptr<ListPositionAllocator> initialPosition = CreateObject<ListPositionAllocator>();
initialPosition->Add(Vector(-240, -240, uav0z));
initialPosition->Add(Vector(240, -240, uav1z));
initialPosition->Add(Vector(-240, 240, uav2z));
initialPosition->Add(Vector(240, 240, uav3z));


  // Apply mobility patterns to UAVs (each helper installs 1 node)
  surveillanceUAV.SetPositionAllocator(initialPosition);
  surveillanceUAV.Install(uavNodes.Get(0));

  patrolUAV.SetPositionAllocator(initialPosition);
  patrolUAV.Install(uavNodes.Get(1));

  rapidUAV.SetPositionAllocator(initialPosition);
  rapidUAV.Install(uavNodes.Get(2));

  strategicUAV.SetPositionAllocator(initialPosition);
  strategicUAV.Install(uavNodes.Get(3));

  // ==================== EPC/NR SETUP ====================
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


  // --- Internet stack on UAVs ---
  InternetStackHelper internet;
  internet.Install(uavNodes);

  Ipv4InterfaceContainer ueIfaces = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

  nrHelper->AttachToClosestGnb(ueDevs, gnbDevs);

  // ==================== Background telemetry from each UAV (Option A) ====================
std::string rateStr = std::to_string(telemetryRateMbps) + "Mbps";


  // ==================== Remote Host (Monitoring Server) ====================
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
std::vector<Ptr<IsacSensingApp>> sensingApps;
sensingApps.reserve(numUAVs);
  for (uint32_t i = 0; i < numUAVs; ++i)
{
OnOffHelper onoff("ns3::UdpSocketFactory",
InetSocketAddress(internetIfaces.GetAddress(1), telemetryPort));

onoff.SetAttribute("DataRate", DataRateValue(DataRate(rateStr)));
onoff.SetAttribute("PacketSize", UintegerValue(telemetryPktSize));

// Keep it continuously ON
onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));

ApplicationContainer apps = onoff.Install(uavNodes.Get(i));
apps.Start(Seconds(1.0 + 0.05*i));
apps.Stop(Seconds(simulationTime - 0.1));
}

//std::ofstream fusedLog("uav_isac_fused.csv");
//fusedLog << "time,cowId,tagId,detected_fused,numUsed,x_fused,y_fused\n";

Ptr<IsacFusionReceiverApp> fusionRx = CreateObject<IsacFusionReceiverApp>();
fusionRx->Configure(port, numUAVs, numCows, snrThresholdDb, nullptr);
remoteHost->AddApplication(fusionRx);
fusionRx->SetStartTime(Seconds(0.0));
fusionRx->SetStopTime(Seconds(simulationTime - 0.5));

  // ==================== UDP server on remote host ====================
  
  // ==================== Telemetry sink on remote host ====================
PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
InetSocketAddress(Ipv4Address::GetAny(), telemetryPort));
ApplicationContainer sinkApps = sinkHelper.Install(remoteHost);
sinkApps.Start(Seconds(0.0));
sinkApps.Stop(Seconds(simulationTime - 0.1));



  // ==================== Cow agent init ====================
  std::vector<CowAgent> cows;
  cows.reserve(numCows);

  Ptr<UniformRandomVariable> urv = CreateObject<UniformRandomVariable>();
  Ptr<UniformRandomVariable> urvAng = CreateObject<UniformRandomVariable>();

  double half = areaSize / 2.0;

  for (uint32_t i = 0; i < numCows; ++i)
  {
    CowAgent c;
    c.id = i;
    c.tagId = 1000 + i;

    c.pos = Vector(urv->GetValue(-0.9*half, 0.9*half),
                   urv->GetValue(-0.9*half, 0.9*half),
                   0.0);

    double ang = urvAng->GetValue(0.0, 2.0*M_PI);
    double spd = urv->GetValue(0.0, cowSpeedMax);
    c.vel = Vector(spd*std::cos(ang), spd*std::sin(ang), 0.0);

    cows.push_back(c);
  }
// Give fusion app access to ground-truth cows for online RMSE
  fusionRx->SetCowTruthSource(&cows);
  // Truth log (optional)
  //std::ofstream cowTruth("cow_truth.csv");
  //cowTruth << "time,cowId,tagId,x,y\n";

  // ISAC report log (optional, per detection)
  //std::ofstream isacLog("uav_isac_reports.csv");
  //isacLog << "time,uavId,cowId,tagId,detected,snr_db,xhat,yhat\n";


  
  // Cow update event
  std::function<void()> updateCows;
  updateCows = [&]() {
    double t = Simulator::Now().GetSeconds();
    for (auto &c : cows)
    {
      // integrate
      c.pos.x += c.vel.x * cowUpdateDt;
      c.pos.y += c.vel.y * cowUpdateDt;

      // reflect boundaries
      if (c.pos.x > half)  { c.pos.x = 2*half - c.pos.x; c.vel.x *= -1; }
      if (c.pos.x < -half) { c.pos.x = -2*half - c.pos.x; c.vel.x *= -1; }
      if (c.pos.y > half)  { c.pos.y = 2*half - c.pos.y; c.vel.y *= -1; }
      if (c.pos.y < -half) { c.pos.y = -2*half - c.pos.y; c.vel.y *= -1; }

      //cowTruth << t << "," << c.id << "," << c.tagId << "," << c.pos.x << "," << c.pos.y << "\n";
    }
    Simulator::Schedule(Seconds(cowUpdateDt), updateCows);
  };
  Simulator::Schedule(Seconds(0.0), updateCows);

  // ==================== Install ISAC sensing apps on UAVs ====================
 for (uint32_t i = 0; i < numUAVs; ++i)
{
  double thisSenseDt = survSenseDt;
  if (i == 1) thisSenseDt = patSenseDt;
  if (i == 2) thisSenseDt = rapidSenseDt;
  if (i == 3) thisSenseDt = stratSenseDt;

  Ptr<IsacSensingApp> app = CreateObject<IsacSensingApp>();
  app->Configure(uavNodes.Get(i), i, &cows,
                 internetIfaces.GetAddress(1), port,
                 Seconds(thisSenseDt),
                 snrThresholdDb,
                 sigmaPosMeters,
                 fc, ptDbm, gDb, rcsDbsm, noiseDbm);
  app->ConfigureProcessingEnergy(kappaProc, c0Proc, c1Proc, fProcHz);
  //app->SetLogFiles(&isacLog);

  uavNodes.Get(i)->AddApplication(app);
  sensingApps.push_back(app);
  app->SetStartTime(Seconds(2.0 + 0.1*i));
  app->SetStopTime(Seconds(simulationTime - 0.5));

  // ---- EXTRA COMM flow: telemetry uplink UAV -> remote host ----
  UdpClientHelper telemetryClient(internetIfaces.GetAddress(1), telemetryPort);
  telemetryClient.SetAttribute("MaxPackets", UintegerValue(0)); // 0 = unlimited in ns-3
  telemetryClient.SetAttribute("Interval", TimeValue(Seconds(telemetryInterval)));
  telemetryClient.SetAttribute("PacketSize", UintegerValue(telemetryPktSize));

  ApplicationContainer telemetryApps = telemetryClient.Install(uavNodes.Get(i));
  telemetryApps.Start(Seconds(1.0 + 0.1*i));
  telemetryApps.Stop(Seconds(simulationTime - 0.5));
}


// ==================== CSV DATA COLLECTION ====================
  

// ==================== ENERGY ACCUMULATION ====================
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

      // Propulsion energy
      double Pprop = PhoverW + cv * speed2;
      ePropJ[i] += Pprop * dt;

      // RF energy (simple approximation using constant UE Tx power)
      eRfJ[i] += PtxUeW * dt;
    }

    Simulator::Schedule(Seconds(dt), sampleEnergy);
  };
// ==================== FlowMonitor KPIs ====================
  FlowMonitorHelper flowHelper;
Ptr<FlowMonitor> monitor = flowHelper.Install(NodeContainer(uavNodes, remoteHostContainer));
Ptr<Ipv4FlowClassifier> classifier =
    DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());

  


  Simulator::Schedule(Seconds(1.0), sampleEnergy);

  // ============================================================
  // xApp bridge — active only when --enableXapp=1 is passed
  // Every 10 sim-seconds:
  //   1. Collect window KPIs from fusionRx
  //   2. Send them to xApp via UDP
  //   3. Check for param updates from xApp
  //   4. Apply any updates to live sensing variables
  // ============================================================
  std::unique_ptr<KpiReporter> kpiReporter;
  if (enableXapp)
  {
    kpiReporter = std::make_unique<KpiReporter>(
        xappHost,
        5555,   // xApp listens here for KPIs from ns-3
        5556    // ns-3 listens here for param updates from xApp
    );
    std::cout << "[xApp] Bridge enabled → "
              << xappHost << ":5555\n";
  }

  // ============================================================
  // KPI tracking CSV — records metrics every 10 sim-seconds
  // ============================================================
  std::ofstream xappKpiLog("xapp_kpi_log.csv");
   xappKpiLog
    << "sim_time_s,"
    // System-level KPIs
    << "pdet,rmse_m,energy_j,jscore,"
    // Per-UAV energy
    << "uav0_energy_j,uav1_energy_j,uav2_energy_j,uav3_energy_j,"
    // Per-UAV propulsion
    << "uav0_eprop_j,uav1_eprop_j,uav2_eprop_j,uav3_eprop_j,"
    // Per-UAV RF energy
    << "uav0_erf_j,uav1_erf_j,uav2_erf_j,uav3_erf_j,"
    // Per-UAV processing energy
    << "uav0_eproc_j,uav1_eproc_j,uav2_eproc_j,uav3_eproc_j,"
    // Communication KPIs
    << "avg_throughput_mbps,avg_delay_ms,avg_loss_pct,"
    // Active parameters
    << "survSenseDt,patSenseDt,rapidSenseDt,stratSenseDt,"
    << "survVmax,patVmax,rapidVmax,stratVmax,"
    << "survZmin,survZmax,patZmin,patZmax,"
    << "rapidZmin,rapidZmax,stratZmin,stratZmax,"
    << "survDt,patDt,rapidDt,stratDt\n";

  std::function<void()> xappCycle;
  xappCycle = [&]() {

    if (enableXapp)
    {
      double simNow = Simulator::Now().GetSeconds();

      double pdet = fusionRx->GetWindowPdet();
      double rmse = fusionRx->GetWindowRmseM();

      double uavEprop[4], uavErf[4], uavEproc[4], uavEtot[4];
      double totalEnergy = 0.0;
      for (uint32_t i = 0; i < numUAVs; ++i)
      {
        uavEprop[i] = ePropJ[i];
        uavErf[i]   = eRfJ[i];
        uavEproc[i] = sensingApps[i]->GetProcessingEnergyJ();
        uavEtot[i]  = uavEprop[i] + uavErf[i] + uavEproc[i];
        totalEnergy += uavEtot[i];
      }

      monitor->CheckForLostPackets();
      auto flowStats = monitor->GetFlowStats();
      double windowThr   = 0.0;
      double windowDelay = 0.0;
      double windowLoss  = 0.0;
      uint32_t nIsacFlows = 0;
      for (const auto& kv : flowStats)
      {
        auto ft = classifier->FindFlow(kv.first);
        if (ft.destinationPort != port) continue;
        auto st = kv.second;
        double dur = (st.timeLastTxPacket -
                      st.timeFirstTxPacket).GetSeconds();
        if (dur > 0.0)
          windowThr += st.rxBytes * 8.0 / dur / 1e6;
        if (st.rxPackets > 0)
          windowDelay += st.delaySum.GetSeconds() /
                         st.rxPackets * 1000.0;
        if (st.txPackets > 0)
          windowLoss += 100.0 *
                        double(st.txPackets - st.rxPackets) /
                        double(st.txPackets);
        nIsacFlows++;
      }
      double avgThr   = (nIsacFlows > 0) ? windowThr   / nIsacFlows : 0.0;
      double avgDelay = (nIsacFlows > 0) ? windowDelay / nIsacFlows : 0.0;
      double avgLoss  = (nIsacFlows > 0) ? windowLoss  / nIsacFlows : 0.0;

      const double thrRefMbps  = 0.0020;
      const double rmseRefM    = 30.0;
      const double delayRefMs  = 20.0;

      double Pdet_n  = ClampToRange(pdet,            0.0, 1.0);
      double Thr_n   = ClampToRange(avgThr/thrRefMbps,  0.0, 1.0);
      double RMSE_n  = ClampToRange(rmse/rmseRefM,   0.0, 1.0);
      double Delay_n = ClampToRange(avgDelay/delayRefMs, 0.0, 1.0);
      double Loss_n  = ClampToRange(avgLoss/100.0,   0.0, 1.0);

      double jscore = 3.0*Pdet_n + 3.0*Thr_n - 2.0*RMSE_n
                    - 1.0*Delay_n - 1.0*Loss_n;

      xappKpiLog
        << simNow      << ","
        << pdet        << ","
        << rmse        << ","
        << totalEnergy << ","
        << jscore      << ","
        << uavEtot[0]  << "," << uavEtot[1]  << ","
        << uavEtot[2]  << "," << uavEtot[3]  << ","
        << uavEprop[0] << "," << uavEprop[1] << ","
        << uavEprop[2] << "," << uavEprop[3] << ","
        << uavErf[0]   << "," << uavErf[1]   << ","
        << uavErf[2]   << "," << uavErf[3]   << ","
        << uavEproc[0] << "," << uavEproc[1] << ","
        << uavEproc[2] << "," << uavEproc[3] << ","
        << avgThr      << ","
        << avgDelay    << ","
        << avgLoss     << ","
        << survSenseDt  << "," << patSenseDt   << ","
        << rapidSenseDt << "," << stratSenseDt << ","
        << survVmax  << "," << patVmax   << ","
        << rapidVmax << "," << stratVmax << ","
        << survZmin  << "," << survZmax  << ","
        << patZmin   << "," << patZmax   << ","
        << rapidZmin << "," << rapidZmax << ","
        << stratZmin << "," << stratZmax << ","
        << survDt  << "," << patDt   << ","
        << rapidDt << "," << stratDt << "\n";
      xappKpiLog.flush();

      kpiReporter->SendKpis(pdet, rmse, totalEnergy, simNow,
                            avgThr, avgDelay, avgLoss);

      std::map<std::string, double> newParams;
      if (kpiReporter->ReceiveParamUpdate(newParams))
      {
        if (newParams.count("survSenseDt"))  survSenseDt  = newParams["survSenseDt"];
        if (newParams.count("patSenseDt"))   patSenseDt   = newParams["patSenseDt"];
        if (newParams.count("rapidSenseDt")) rapidSenseDt = newParams["rapidSenseDt"];
        if (newParams.count("stratSenseDt")) stratSenseDt = newParams["stratSenseDt"];
        if (newParams.count("survVmax"))     survVmax     = newParams["survVmax"];
        if (newParams.count("patVmax"))      patVmax      = newParams["patVmax"];
        if (newParams.count("rapidVmax"))    rapidVmax    = newParams["rapidVmax"];
        if (newParams.count("stratVmax"))    stratVmax    = newParams["stratVmax"];
        if (newParams.count("survZmin"))     survZmin     = newParams["survZmin"];
        if (newParams.count("survZmax"))     survZmax     = newParams["survZmax"];
        if (newParams.count("patZmin"))      patZmin      = newParams["patZmin"];
        if (newParams.count("patZmax"))      patZmax      = newParams["patZmax"];
        if (newParams.count("rapidZmin"))    rapidZmin    = newParams["rapidZmin"];
        if (newParams.count("rapidZmax"))    rapidZmax    = newParams["rapidZmax"];
        if (newParams.count("stratZmin"))    stratZmin    = newParams["stratZmin"];
        if (newParams.count("stratZmax"))    stratZmax    = newParams["stratZmax"];
        if (newParams.count("survDt"))       survDt       = newParams["survDt"];
        if (newParams.count("patDt"))        patDt        = newParams["patDt"];
        if (newParams.count("rapidDt"))      rapidDt      = newParams["rapidDt"];
        if (newParams.count("stratDt"))      stratDt      = newParams["stratDt"];

        sensingApps[0]->SetSensingInterval(survSenseDt);
        sensingApps[1]->SetSensingInterval(patSenseDt);
        sensingApps[2]->SetSensingInterval(rapidSenseDt);
        sensingApps[3]->SetSensingInterval(stratSenseDt);

        std::cout << "[xApp] Param update applied at t="
                  << simNow << "s\n";
      }

      fusionRx->ResetWindowStats();
    }

    Simulator::Schedule(Seconds(10.0), xappCycle);
  };

  Simulator::Schedule(Seconds(20.0), xappCycle);

  Simulator::Stop(Seconds(simulationTime));
  Simulator::Run();

  // KPI print
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

// Only accumulate ISAC flows into your objective metrics
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

double ePropTot = 0.0, eRfTot = 0.0, eProcTot = 0.0, eTotTot = 0.0;

for (uint32_t i = 0; i < numUAVs; ++i)
{
  ePropTot += ePropJ[i];
  eRfTot += eRfJ[i];
  eProcTot += sensingApps[i]->GetProcessingEnergyJ();
}

std::cout << "\n===== PER-UAV ENERGY =====\n";

for (uint32_t i = 0; i < numUAVs; ++i)
{
  double eProc = sensingApps[i]->GetProcessingEnergyJ();
  double eTot = ePropJ[i] + eRfJ[i] + eProc;

  std::cout << "UAV " << i
            << " | Eprop=" << ePropJ[i]
            << " J | ERF=" << eRfJ[i]
            << " J | Eproc=" << eProc
            << " J | Etot=" << eTot
            << " J\n";
}

eTotTot = ePropTot + eRfTot + eProcTot;
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


// append summary row
// append summary row
bool needHeader = FileEmptyOrMissing("run_summary_xapp.csv");
std::ofstream runSummary("run_summary_xapp.csv", std::ios::app);

if (needHeader)
{
  runSummary
<< "rngRun,numCows,snrThDb,sigmaPosM,"
<< "ueTxPowerDbm,gnbTxPowerDbm,cv,"
<< "survSenseDt,patSenseDt,rapidSenseDt,stratSenseDt,"
<< "Pdet,RMSEm,Jfinal,"
<< "alphaSurv,alphaPat,alphaRapid,alphaStrat,"
<< "survVmin,survVmax,patVmin,patVmax,rapidVmin,rapidVmax,stratVmin,stratVmax,"
<< "survZmin,survZmax,patZmin,patZmax,rapidZmin,rapidZmax,stratZmin,stratZmax,"
<< "survDt,patDt,rapidDt,stratDt,"
<< "avgThrMbps,avgDelayMs,avgLossPct,"
<< "UAV0_Energy_J,UAV1_Energy_J,UAV2_Energy_J,UAV3_Energy_J,"
<< "EpropJ,ERFJ,EprocJ,EtotJ\n";
}

double uav0Etot = (numUAVs > 0) ? (ePropJ[0] + eRfJ[0] + sensingApps[0]->GetProcessingEnergyJ()) : 0.0;
double uav1Etot = (numUAVs > 1) ? (ePropJ[1] + eRfJ[1] + sensingApps[1]->GetProcessingEnergyJ()) : 0.0;
double uav2Etot = (numUAVs > 2) ? (ePropJ[2] + eRfJ[2] + sensingApps[2]->GetProcessingEnergyJ()) : 0.0;
double uav3Etot = (numUAVs > 3) ? (ePropJ[3] + eRfJ[3] + sensingApps[3]->GetProcessingEnergyJ()) : 0.0;

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
  << eTotTot
  << "\n";

runSummary.close();



  

  //cowTruth.close();
  //isacLog.close();
  //fusedLog.close();

xappKpiLog.close();
  Simulator::Destroy();
  std::cout << "\nFINAL_PDET=" << runPdet << "\n";
  std::cout << "FINAL_RMSE=" << runRmse << "\n";
  std::cout << "FINAL_J=" << Jfinal << "\n";
  std::cout << "\nDone.\n";
  std::cout << "Generated: cow_truth.csv, uav_isac_reports.csv, uav_isac_fused.csv\n";

  return 0;
}
