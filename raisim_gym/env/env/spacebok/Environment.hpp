//
// Created by jemin on 3/27/19.
// MIT License
//
// Copyright (c) 2019-2019 Robotic Systems Lab, ETH Zurich
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/* Convention
*
*   observation space = [ height                                                      n =  1, si =  0
*                         z-axis in world frame expressed in body frame (R_b.row(2))  n =  3, si =  1
 *                        target forward vel                                          n = 1
*                         joint angles,                                               n =  8, si =  4
*                         body Linear velocities,                                     n =  3, si = 16
*                         body Angular velocities,                                    n =  3, si = 19
*                         joint velocities,                                           n =  8, si = 22
 *                        previous torques                                            n = 8] total 35
*
*/


#include <stdlib.h>
#include <cstdint>
#include <set>
#include <raisim/OgreVis.hpp>
#include "RaisimGymEnv.hpp"
#include "visSetupCallback.hpp"

#include "visualizer/raisimKeyboardCallback.hpp"
#include "visualizer/helper.hpp"
#include "visualizer/guiState.hpp"
#include "visualizer/raisimBasicImguiPanel.hpp"


namespace raisim {

  class ENVIRONMENT : public RaisimGymEnv {

  public:

    explicit ENVIRONMENT(const std::string &resourceDir, const YAML::Node &cfg, bool visualizable) :
        RaisimGymEnv(resourceDir, cfg), distribution_(0.0, 0.2), visualizable_(visualizable) {

      /// add objects
      spacebok_ = world_->addArticulatedSystem(resourceDir_ + "/urdf/spacebok.urdf");
      spacebok_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
      auto ground = world_->addGround();
      groundIndex_ = ground->getIndexInWorld();
      world_->setERP(0, 0);

      /// get robot data
      gcDim_ = spacebok_->getGeneralizedCoordinateDim();
      gvDim_ = spacebok_->getDOF();
      nJoints_ = 8;

      /// initialize containers
      gc_.setZero(gcDim_);
      gc_init_.setZero(gcDim_);
      gv_.setZero(gvDim_);
      gv_init_.setZero(gvDim_);
      torque_.setZero(gvDim_);
      pTarget_.setZero(gcDim_);
      vTarget_.setZero(gvDim_);
      pTarget8_.setZero(nJoints_);
      previousTorques_.setZero(nJoints_);

      /// this is a good standing  configuration of spacebok
      gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, -0.45, 0.4, -0.45, 0.4, 0.05, 0.45, 0.05, 0.45;

      /// set pd gains
      Eigen::VectorXd upper = Eigen::VectorXd::Constant(gvDim_, 25.);
      Eigen::VectorXd lower = Eigen::VectorXd::Constant(gvDim_, -25.);
      spacebok_->setActuationLimits(upper, lower);
      Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
      jointPgain.setZero();
      jointPgain.tail(nJoints_).setConstant(60.0);
      jointDgain.setZero();
      jointDgain.tail(nJoints_).setConstant(3.0);
      spacebok_->setPdGains(jointPgain, jointDgain);
      spacebok_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

      /// MUST BE DONE FOR ALL ENVIRONMENTS
      obDim_ = 27; /// convention described on top
      actionDim_ = nJoints_;
      actionMean_.setZero(actionDim_);
      actionStd_.setZero(actionDim_);
      obMean_.setZero(obDim_);
      obStd_.setZero(obDim_);

      /// action & observation scaling
      actionMean_ = gc_init_.tail(nJoints_);
      actionStd_.setConstant(0.6);

      obMean_ << 0.44, /// average height
          0.0, 0.0, 0.0, /// gravity axis 3
          0.,
          gc_init_.tail(nJoints_), /// joint position
          Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
          Eigen::VectorXd::Constant(nJoints_, 0.0); /// joint vel
//          Eigen::VectorXd::Constant(nJoints_, 0.0); /// previous torques

      obStd_ << 0.1, /// average height
          Eigen::VectorXd::Constant(3, 0.7), /// gravity axes angles
          1.,
          Eigen::VectorXd::Constant(nJoints_, 1.), /// joint angles
          Eigen::VectorXd::Constant(3, 2.), /// linear velocity
          Eigen::VectorXd::Constant(3, 4.), /// angular velocities
          Eigen::VectorXd::Constant(nJoints_, 10.); /// joint velocities
//          Eigen::VectorXd::Constant(nJoints_, 20.); /// previous torques

      /// Reward coefficients
      READ_YAML(double, forwardVelRewardCoeff_, cfg["forwardVelRewardCoeff"])
      READ_YAML(double, torqueRewardCoeff_, cfg["torqueRewardCoeff"])
      READ_YAML(double, clearanceRewardCoeff_, cfg["clearanceRewardCoeff"])
      READ_YAML(double, slipRewardCoeff_, cfg["slipRewardCoeff"])
      READ_YAML(double, orientRewardCoeff_, cfg["orientRewardCoeff"])
      READ_YAML(double, gaitRewardCoeff_, cfg["gaitRewardCoeff"])
      READ_YAML(double, globalRewardScale_, cfg["globalRewardScale"])
      /// parameters
      READ_YAML(bool, verbose_, cfg["verbose"])
      READ_YAML(double, gravity_, cfg["gravity"])
      READ_YAML(double, maxTime_, cfg["max_time"])

      world_->setGravity({0., 0., gravity_});

      gui::rewardLogger.init({"forwardVelReward", "torqueReward", "clearanceReward", "slipReward"});

      /// indices of links that can make contact with ground
      feetIndices_.push_back(spacebok_->getBodyIdx("lower_leg_long_front_left"));
      feetIndices_.push_back(spacebok_->getBodyIdx("lower_leg_long_front_right"));
      feetIndices_.push_back(spacebok_->getBodyIdx("lower_leg_long_hind_left"));
      feetIndices_.push_back(spacebok_->getBodyIdx("lower_leg_long_hind_right"));
      foot_offset_ = {-0.25, 0., 0.};

      /// ignore collisions between parents and child links
//    spacebok_->ignoreCollisionBetween(0, 1);
//    spacebok_->ignoreCollisionBetween(0, 3);
//    spacebok_->ignoreCollisionBetween(0, 5);
//    spacebok_->ignoreCollisionBetween(0, 7);
//    spacebok_->ignoreCollisionBetween(1, 2);
//    spacebok_->ignoreCollisionBetween(3, 4);
//    spacebok_->ignoreCollisionBetween(5, 6);
//    spacebok_->ignoreCollisionBetween(7, 8);
//    for(std::string body:spacebok_->getBodyNames())
//        std::cout << body << std::endl;
      /// visualize if it is the first environment
      if (visualizable_) {
        auto vis = raisim::OgreVis::get();

        /// these method must be called before initApp
        vis->setWorld(world_.get());
        vis->setWindowSize(1280, 720);
        vis->setImguiSetupCallback(imguiSetupCallback);
        vis->setImguiRenderCallback(imguiRenderCallBack);
        vis->setKeyboardCallback(raisimKeyboardCallback);
        vis->setSetUpCallback(setupCallback);
        vis->setAntiAliasing(2);

        /// starts visualizer thread
        vis->initApp();

        visual_ = vis->createGraphicalObject(spacebok_, "spacebok");
        vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");
        desired_fps_ = 60.;
        vis->setDesiredFPS(desired_fps_);
        resetCamera();
      }
    }

    ~ENVIRONMENT() final = default;

    void init() final {}

    void reset() final {
//    spacebok_->setState(gc_init_, gv_init_);
      resetToRandomState();
      updateObservation();
      totalForwardVelReward_ = 0.;
      totalTorqueReward_ = 0.;
      totalClearanceReward_ = 0.;
      totalSlipReward_ = 0.;
      totalOrientReward_ = 0.;
      totalGaitReward_ = 0.;
      targetVelx_ = sampleUniformInt(-2, 2) / 2.;
      collision_ = false;
      episodeTime_ = 0;
      if (visualizable_) {
        gui::rewardLogger.clean();
        resetCamera();
      }
    }

    void resetCamera() {
      auto vis = raisim::OgreVis::get();
      vis->select(visual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, false);
    }

    float step(const Eigen::Ref<EigenVec> &action) final {
      /// action scaling
      pTarget8_ = action.cast<double>();
      pTarget8_ = pTarget8_.cwiseProduct(actionStd_);
      pTarget8_ += actionMean_;
      pTarget_.tail(nJoints_) = pTarget8_;
      spacebok_->setPdTarget(pTarget_, vTarget_);

      episodeTime_ += control_dt_;
      auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
      auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);

      for (int i = 0; i < loopCount; i++) {
        world_->integrate();

        if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
          raisim::OgreVis::get()->renderOneFrame();

        visualizationCounter_++;
      }

      if(episodeTime_ >= maxTime_ / 2 and episodeTime_ < maxTime_ / 2 + control_dt_){
        targetVelx_ = sampleUniformInt(-2, 2) / 2.;
      }

      updateObservation();
      updateContacts();

      Eigen::VectorXd torques = spacebok_->getGeneralizedForce().e().tail(nJoints_);
      torqueReward_ = -torqueRewardCoeff_ * torques.squaredNorm() / globalRewardScale_;
      torqueReward_ -= 2*torqueRewardCoeff_*(previousTorques_ - torques).squaredNorm() / globalRewardScale_;
      previousTorques_ = torques;
//    torques = torques.array().abs().max(10.) - 10.;
//    torqueReward_ -= 5*torqueRewardCoeff_ * torques.matrix().squaredNorm() / globalRewardScale_;

//    forwardVelReward_ = forwardVelRewardCoeff_ *
//                        std::exp(-std::pow((0.5 - bodyLinearVel_[0])/0.2, 2)) / globalRewardScale_;
//      forwardVelReward_ = -forwardVelRewardCoeff_ *
//                          (std::pow((targetVelx_ - bodyLinearVel_[0]), 2) + 0.25 * std::pow((0. - bodyLinearVel_[2]), 2)) /
//                          globalRewardScale_;
      forwardVelReward_ = forwardVelRewardCoeff_ * std::exp(-(std::pow((targetVelx_ - bodyLinearVel_[0]), 2) +
                                                                 0. * std::pow((0. - bodyLinearVel_[2]), 2)) / .5) /
                                                    globalRewardScale_;
//    forwardVelReward_ = forwardVelRewardCoeff_ * bodyLinearVel_[0] / globalRewardScale_;
      slipReward_ = -slipRewardCoeff_ * (feetContacts_ == -1).count() / globalRewardScale_;
      orientReward_ = -orientRewardCoeff_ * (std::pow(measuredGravityVector_[0], 2) +
                                             std::pow(measuredGravityVector_[1], 2)) / globalRewardScale_;
//      gaitReward_ = -gaitRewardCoeff_ * ((action.segment(0, 2)
//                                          - action.segment(6, 2)).array().abs().sum() +
//                                         (action.segment(2, 2)
//                                          - action.segment(4, 2)).array().abs().sum())
//                                        / globalRewardScale_;

        gaitReward_ = -gaitRewardCoeff_ * gv_.tail(nJoints_).squaredNorm() / globalRewardScale_;
      clearanceReward_ = 0.;
      for (int i = 0; i < feetIndices_.size(); i++) {
        if (feetContacts_[i] != 0)
          continue;
        int foot_index = feetIndices_[i];
        raisim::Vec<3> foot_pos{};
        raisim::Vec<3> foot_vel{};
        raisim::Mat<3, 3> rot;
//      spacebok_->getPosition_W(foot_index,foot_offset_,foot_pos);
        spacebok_->getBodyOrientation(foot_index, rot);
        spacebok_->getBodyPosition(foot_index, foot_pos);
        foot_pos.e() += rot.e() * foot_offset_.e();
        spacebok_->getVelocity(foot_index, foot_offset_, foot_vel);
        foot_vel[2] = 0.;
        clearanceReward_ -= clearanceRewardCoeff_ * std::abs(0.08 - foot_pos[2]) * foot_vel.norm();
        if (verbose_) std::cout << "foot vel x: " << foot_vel[0] << std::endl;
      }
      clearanceReward_ /= globalRewardScale_;

      if (visualizeThisStep_) {
        gui::rewardLogger.log("torqueReward", torqueReward_);
        gui::rewardLogger.log("forwardVelReward", forwardVelReward_);
        gui::rewardLogger.log("clearanceReward", clearanceReward_);
        gui::rewardLogger.log("slipReward", slipReward_);
        gui::rewardLogger.log("orientReward", orientReward_);
        gui::rewardLogger.log("gaitReward", gaitReward_);
      }

      totalForwardVelReward_ += forwardVelReward_;
      totalTorqueReward_ += torqueReward_;
      totalClearanceReward_ += clearanceReward_;
      totalSlipReward_ += slipReward_;
      totalOrientReward_ += orientReward_;
      totalGaitReward_ += gaitReward_;
      return torqueReward_ + forwardVelReward_ + clearanceReward_ + slipReward_ + orientReward_ + gaitReward_;
    }

    void updateExtraInfo() final {
      extraInfo_["ep_forward_vel_reward"] = totalForwardVelReward_;
      extraInfo_["ep_torque_reward"] = totalTorqueReward_;
      extraInfo_["ep_clearance_reward"] = totalClearanceReward_;
      extraInfo_["ep_slip_reward"] = totalSlipReward_;
      extraInfo_["ep_orient_reward"] = totalOrientReward_;
      extraInfo_["ep_gait_reward"] = totalGaitReward_;
    }

    void updateObservation() {
      spacebok_->getState(gc_, gv_);
      obDouble_.setZero(obDim_);
      obScaled_.setZero(obDim_);

      /// body height
      obDouble_[0] = gc_[2];

      /// body orientation
      raisim::Vec<4> quat{};
      raisim::Mat<3, 3> rot;
      quat[0] = gc_[3];
      quat[1] = gc_[4];
      quat[2] = gc_[5];
      quat[3] = gc_[6];
      raisim::quatToRotMat(quat, rot);
      measuredGravityVector_ = rot.e().row(2);
      obDouble_.segment(1, 3) = measuredGravityVector_;
      obDouble_(4)  = targetVelx_;

      /// joint angles
      obDouble_.segment(5, nJoints_) = gc_.tail(nJoints_);

      /// body velocities
      bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
      bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
      obDouble_.segment(13, 3) = bodyLinearVel_;
      obDouble_.segment(16, 3) = bodyAngularVel_;

      /// joint velocities
      obDouble_.segment(19, nJoints_) = gv_.tail(nJoints_);

      /// previous torques
//      obDouble_.segment(27, nJoints_) = previousTorques_;

      /// normalize
      obScaled_ = (obDouble_ - obMean_).cwiseQuotient(obStd_);
    }

    void resetToRandomState() {
      double d_height = sampleUniform(0., 0.2);
      raisim::Vec<3> rpy{sampleUniform(-0.2, 0.2),
                         sampleUniform(-0.2, 0.2),
                         sampleUniform(-M_PI, M_PI)};
      raisim::Vec<4> quat{};
      raisim::eulerVecToQuat(rpy, quat);
      Eigen::VectorXd d_joint_pos(nJoints_);
      for (int i = 0; i < nJoints_; i++) {
        d_joint_pos(i) = sampleUniform(-0.25, 0.25);
      }
      gc_ = gc_init_;
      gc_(2) += d_height;
      gc_.segment(3, 4) = quat.e();
      gc_.tail(nJoints_) += d_joint_pos;
      spacebok_->setState(gc_, gv_init_);
    }

    void updateContacts() {
      feetContacts_.setZero();
      auto activeContacts = spacebok_->getContacts();
      for (size_t c = 0; c < spacebok_->getContacts().size(); c++) {
        const auto &contact = activeContacts[c];
        if (contact.skip()) {
          continue;
        }
        size_t foot = -1;
        size_t idx = contact.getlocalBodyIndex();
        for (size_t k = 0; k < 4; k++) {
          if (feetIndices_[k] == idx)
            foot = k;
        }
        if (foot == -1) {
          collision_ = true;
          return;
        }
        raisim::Vec<3> vel;
        raisim::Vec<3> pos;
        spacebok_->getBodyPosition(idx, pos);
        if ((contact.getPosition().e() - pos.e()).norm() < 0.2) {
          collision_ = true;
          return;
        }
        spacebok_->getContactPointVel(c, vel);
//        feetContactVelocities_[k] = vel.e();
        if (vel.e().head(2).norm() > 0.001) {
          feetContacts_[foot] = -1;
          if (verbose_) std::cout << "slip foot" << foot << std::endl;
        } else {
          feetContacts_[foot] = 1;
          if (verbose_) std::cout << "contact" << foot << std::endl;
        }
      }
    }

    void observe(Eigen::Ref<EigenVec> ob) final {
      /// convert it to float
      ob = obScaled_.cast<float>();
    }

    bool isTerminalState(float &terminalReward) final {
      terminalReward = float(terminalRewardCoeff_);
//    /// check that joint limits are respected
      auto joints = gc_.tail(nJoints_);
      if (collision_)
        return true;
      for (int i = 0; i < 4; i++) {
        double j1 = joints(2 * i) + M_PI / 4.;
        double j2 = joints(2 * i + 1) + M_PI / 2.;
        if (j1 > M_PI / 2. or j1 < -M_PI / 2.) {
//        std::cout << "j1 out of bounds: " << j1 << std::endl;
          return true;
        }
        if (j2 > (5. / 6.) * M_PI or j2 < M_PI / 6.) {
//        std::cout << "j2 out of bounds: " << j2 << std::endl;
          return true;
        }
      }
      terminalReward = 0.;
      return episodeTime_ > maxTime_;
    }

    void processCall(std::unordered_map<std::string, double> call) {
      raisim::Vec<3> rpy{};
      raisim::Vec<3> vel_lin{};
      raisim::Vec<3> vel_ang{};
      double base_heigt = 0;
      if (call.find("gravity") != call.end()) {
        world_->setGravity({0, 0, call["gravity"]});
      }
      else if (call.find("set_verbose") != call.end()){
        verbose_ = not verbose_;
      }
      else if (call.find("target_vel_x") != call.end()){
        targetVelx_ = call["target_vel_x"];
        std::cout << targetVelx_;
      }
      else if (call.find("reset_state") != call.end()) {
        reset();
        rpy[0] = call["roll"];
        rpy[1] = call["pitch"];
        rpy[2] = call["yaw"];
        raisim::Vec<4> quat{};
        raisim::eulerVecToQuat(rpy, quat);
        gc_ = gc_init_;
        gc_[2] = call["base_height"];
        gc_.segment(3, 4) = quat.e();

        vel_lin[0] = call["vel_lin_x"];
        vel_lin[1] = call["vel_lin_y"];
        vel_lin[2] = call["vel_lin_z"];
        vel_ang[0] = call["vel_ang_u"];
        vel_ang[1] = call["vel_ang_v"];
        vel_ang[2] = call["vel_ang_w"];
        gv_ = gv_init_;
        gv_.segment(0, 6) << vel_lin.e(), vel_ang.e();

        spacebok_->setState(gc_, gv_);
      }
    }

    void setSeed(int seed) final {
      std::srand(seed);
    }

    double sampleUniform(double a, double b) {
      return a + std::rand() / (RAND_MAX + 1.) * (b - a);
    }

    int sampleUniformInt(double a, double b){
      return std::round(sampleUniform(a - 0.5,b + 0.5));
    }

    void close() final {
    }

  private:
    bool verbose_ = false;
    double episodeTime_ = 0., maxTime_ = 0.;
    double gravity_ = -9.8107;
    int gcDim_, gvDim_, nJoints_;
    bool visualizable_ = false;
    std::normal_distribution<double> distribution_;
    raisim::ArticulatedSystem *spacebok_;
    int groundIndex_;
    std::vector<GraphicObject> *visual_;
    Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget8_, vTarget_, torque_;
    double terminalRewardCoeff_ = -10.;
    double forwardVelRewardCoeff_ = 0., forwardVelReward_ = 0., totalForwardVelReward_ = 0.;
    double torqueRewardCoeff_ = 0., torqueReward_ = 0., totalTorqueReward_;
    double clearanceRewardCoeff_ = 0., clearanceReward_ = 0., totalClearanceReward_ = 0.;
    double slipRewardCoeff_ = 0., slipReward_ = 0., totalSlipReward_ = 0.;
    double orientRewardCoeff_ = 0., orientReward_ = 0., totalOrientReward_ = 0.;
    double gaitRewardCoeff_ = 0., gaitReward_ = 0., totalGaitReward_ = 0.;
    double globalRewardScale_ = 1.;
    double desired_fps_ = 60.;
    int visualizationCounter_ = 0;
    bool collision_ = false;
    double targetVelx_ = 0.5;
    Eigen::Array4i feetContacts_;
    Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
    Eigen::VectorXd obDouble_, obScaled_;
    Eigen::VectorXd previousTorques_;
    Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
    Eigen::Vector3d measuredGravityVector_;
    std::vector<size_t> feetIndices_;
    raisim::Vec<3> foot_offset_;
  };

}

