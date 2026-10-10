// shoehorn-hwmon: reads the temperature sensors through LibreHardwareMonitorLib every 2 s and serves them as
// /data.json on 127.0.0.1 (port 8085, or the first argument), in the tree shape of LibreHardwareMonitor's own web
// server: root > computer > hardware > sensors, each sensor with SensorId, Type, Text and Value. Sub-hardware sensors
// are listed under their hardware. Only temperatures are served; nothing listens beyond this machine.

using System.Globalization;
using System.Net;
using System.Text.Json;
using LibreHardwareMonitor.Hardware;

int port = args.Length > 0 && int.TryParse(args[0], out var p) ? p : 8085;

var computer = new Computer
{
    IsCpuEnabled = true,
    IsGpuEnabled = true,
    IsMemoryEnabled = true,
    IsMotherboardEnabled = true,
    IsStorageEnabled = true,
    IsControllerEnabled = true,
};
computer.Open();

byte[] body = "{}"u8.ToArray();
var gate = new object();

void Sample()
{
    var hardware = new List<object>();
    foreach (IHardware hw in computer.Hardware)
    {
        hw.Update();
        var sensors = new List<object>();
        void Add(IHardware h)
        {
            foreach (ISensor s in h.Sensors)
            {
                if (s.SensorType != SensorType.Temperature || s.Value is not float v) continue;
                sensors.Add(new Dictionary<string, object>
                {
                    ["Text"] = s.Name,
                    ["SensorId"] = s.Identifier.ToString(),
                    ["Type"] = "Temperature",
                    ["Value"] = v.ToString("0.0", CultureInfo.InvariantCulture) + " C",
                });
            }
        }
        Add(hw);
        foreach (IHardware sub in hw.SubHardware)
        {
            sub.Update();
            Add(sub);
        }
        hardware.Add(new Dictionary<string, object> { ["Text"] = Clean(hw.Name), ["Children"] = sensors });
    }
    var root = new Dictionary<string, object>
    {
        ["Text"] = "Sensor",
        ["Children"] = new[] { new Dictionary<string, object> { ["Text"] = Environment.MachineName, ["Children"] = hardware } },
    };
    byte[] json = JsonSerializer.SerializeToUtf8Bytes(root);
    lock (gate) body = json;
}

// names read from hardware (a memory module's SPD part number) can carry unprintable bytes
static string Clean(string s) =>
    System.Text.RegularExpressions.Regex.Replace(new string(s.Where(c => c >= ' ' && c <= '~').ToArray()), " {2,}", " ").Trim();

Sample();
var timer = new Timer(_ => { try { Sample(); } catch { } }, null, 2000, 2000);

var listener = new HttpListener();
listener.Prefixes.Add($"http://127.0.0.1:{port}/");
listener.Start();
while (true)
{
    HttpListenerContext ctx = listener.GetContext();
    try
    {
        byte[] b;
        lock (gate) b = body;
        if (ctx.Request.Url?.AbsolutePath == "/data.json")
        {
            ctx.Response.ContentType = "application/json";
            ctx.Response.OutputStream.Write(b);
        }
        else
        {
            ctx.Response.StatusCode = 404;
        }
    }
    catch { }
    finally { ctx.Response.Close(); }
}
